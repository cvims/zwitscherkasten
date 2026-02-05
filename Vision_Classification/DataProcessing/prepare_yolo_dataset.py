#!/usr/bin/env python3
"""
Species dataset -> (1) YOLO Detection Dataset + (2) ImageNet-style Crops Dataset.

Input (ImageFolder-style):
  data/<species>/*.(jpg|png|webp)

Outputs:
1) YOLO detect dataset:
  data/iNat_species_yolo_det/
    images/{train,val,test}/<species>/<image>.jpg
    labels/{train,val,test}/<species>/<image>.txt
  data/iNat_species_yolo_det.yaml

2) Crops dataset (ImageNet-style, for MobileNet classifier):
  data/iNat_species_crops_cls/
    train/<species>/<image>.jpg
    val/<species>/<image>.jpg
    test/<species>/<image>.jpg

Method:
- Use pretrained YOLO COCO detector (e.g., yolo11l.pt), keep only COCO class 'bird' (idx=14)
- For each image: take best bird box (highest conf), assign the image's species label to that box
- Split is STRATIFIED by species, and ONLY includes images where a bird was detected
- Save:
  (a) full image + YOLO label
  (b) cropped bird jpg for classifier training

Notes:
- Noisy if multiple birds/species exist; we keep only top-1 bird box.
"""

from __future__ import annotations

import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

# --------------------
# Config
# --------------------
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
DEV = "mps" if torch.backends.mps.is_available() else ("0" if torch.cuda.is_available() else "cpu")

IN_ROOT = Path("data/iNat_images")  # expects data/<class>/<image>

# Output roots
OUT_YOLO = Path("data/iNat_species_yolo_det")        # full images + labels
OUT_CROPS = Path("data/iNat_species_crops_cls")      # crops in ImageNet format
YAML_OUT = Path("data/iNat_species_yolo_det.yaml")

DET_WEIGHTS = "yolo11l.pt"
BIRD_IDX = 14

CONF = 0.05
IOU = 0.5
IMGSZ = 960
PAD = 0.12
MIN_AREA_FRAC = 0.001

TRAIN_FRAC = 0.80
VAL_FRAC = 0.10
TEST_FRAC = 0.10
SEED = 7

# Crop settings (classifier)
CROP_MAX_SIDE = 1024   # resize down if crop is huge; 0/None disables
CROP_JPEG_QUALITY = 92

DEBUG_DIR = Path("debug_skips")
SAVE_NO_BIRD = 50

EXTS = {".jpg", ".jpeg", ".png", ".webp"}
random.seed(SEED)


# --------------------
# Helpers
# --------------------
def safe_copy_as_jpg(src: Path, dst: Path) -> None:
    """
    Copy image into dst, converting to JPG if needed.
    Ensures YOLO images are stored as JPG (stable + smaller).
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    img = Image.open(src).convert("RGB")
    img.save(dst, quality=92, subsampling=0)


def safe_copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    shutil.copy2(src, dst)

def ensure_split_class_dirs(root: Path, split: str, class_names: list[str]) -> None:
    # crops (ImageNet)
    for c in class_names:
        (OUT_CROPS / split / c).mkdir(parents=True, exist_ok=True)
    # yolo images + labels
    for c in class_names:
        (OUT_YOLO / "images" / split / c).mkdir(parents=True, exist_ok=True)
        (OUT_YOLO / "labels" / split / c).mkdir(parents=True, exist_ok=True)



def pad_box_to_yolo_xywh(xyxy: np.ndarray, w: int, h: int, pad: float) -> tuple[float, float, float, float]:
    """xyxy in pixel -> padded, clipped -> normalized xywh for YOLO label."""
    x1, y1, x2, y2 = xyxy.astype(float)
    pw, ph = (x2 - x1) * pad, (y2 - y1) * pad
    x1, y1 = max(0.0, x1 - pw), max(0.0, y1 - ph)
    x2, y2 = min(float(w), x2 + pw), min(float(h), y2 + ph)
    bw, bh = max(0.0, x2 - x1), max(0.0, y2 - y1)
    xc, yc = x1 + bw / 2.0, y1 + bh / 2.0
    return xc / w, yc / h, bw / w, bh / h


def crop_pad_pil(img: Image.Image, xyxy: np.ndarray, pad: float) -> Image.Image:
    w, h = img.size
    x1, y1, x2, y2 = xyxy.astype(float)
    pw, ph = (x2 - x1) * pad, (y2 - y1) * pad
    x1, y1 = int(max(0, x1 - pw)), int(max(0, y1 - ph))
    x2, y2 = int(min(w, x2 + pw)), int(min(h, y2 + ph))
    return img.crop((x1, y1, x2, y2))


def maybe_resize_max_side(img: Image.Image, max_side: int | None) -> Image.Image:
    if not max_side or max_side <= 0:
        return img
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    s = max_side / m
    return img.resize((int(w * s), int(h * s)))


def best_bird_box(res) -> tuple[np.ndarray, float] | None:
    b = res.boxes
    if b is None or len(b) == 0:
        return None
    cls = b.cls.detach().cpu().numpy().astype(int)
    conf = b.conf.detach().cpu().numpy().astype(float)
    keep = np.where(cls == BIRD_IDX)[0]
    if keep.size == 0:
        return None
    i = keep[np.argmax(conf[keep])]
    return b.xyxy[i].detach().cpu().numpy(), float(conf[i])


def list_items(root: Path) -> list[tuple[Path, str]]:
    ignore = {OUT_YOLO.name, OUT_CROPS.name, DEBUG_DIR.name}
    items: list[tuple[Path, str]] = []
    for cls_dir in root.iterdir():
        if not cls_dir.is_dir():
            continue
        if cls_dir.name in ignore:
            continue
        cls = cls_dir.name
        for p in cls_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in EXTS:
                items.append((p, cls))
    return items


def stratified_split(
    items: list[tuple[Path, str]],
    train_frac: float,
    val_frac: float,
    test_frac: float,
) -> tuple[list[tuple[Path, str]], list[tuple[Path, str]], list[tuple[Path, str]]]:
    by_cls: dict[str, list[Path]] = defaultdict(list)
    for p, c in items:
        by_cls[c].append(p)

    train_out: list[tuple[Path, str]] = []
    val_out: list[tuple[Path, str]] = []
    test_out: list[tuple[Path, str]] = []

    for c, ps in by_cls.items():
        random.shuffle(ps)
        n = len(ps)
        if n == 0:
            continue

        if n == 1:
            train_out.append((ps[0], c))
            continue

        if n == 2:
            val_out.append((ps[0], c))
            train_out.append((ps[1], c))
            continue

        test_out.append((ps[0], c))
        val_out.append((ps[1], c))
        for p in ps[2:]:
            train_out.append((p, c))

    random.shuffle(train_out)
    random.shuffle(val_out)
    random.shuffle(test_out)
    return train_out, val_out, test_out


def write_yaml(names: list[str]) -> None:
    YAML_OUT.write_text(
        "path: " + OUT_YOLO.as_posix() + "\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)),
        encoding="utf-8",
    )


# --------------------
# Main
# --------------------
def main() -> None:
    if not IN_ROOT.is_dir():
        raise FileNotFoundError(f"Missing input root: {IN_ROOT}")

    items = list_items(IN_ROOT)
    if not items:
        raise RuntimeError(f"No images found under: {IN_ROOT} (expected data/<class>/*.jpg)")

    # class list / ids from directory names
    names = sorted({c for _p, c in items})
    cls2id = {n: i for i, n in enumerate(names)}

    det = YOLO(DET_WEIGHTS)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    reasons = Counter()
    saved_no_bird = 0

    # First pass: detect birds, keep only images with bird, store per-image meta
    kept_records: list[tuple[Path, str, str, np.ndarray]] = []


    bar = tqdm(items, desc="Detect birds (collect kept images)", unit="img")
    for src, cls in bar:
        try:
            img = Image.open(src).convert("RGB")
        except Exception:
            reasons["open_fail"] += 1
            continue

        r = det.predict(
            img, device=DEV, conf=CONF, iou=IOU, imgsz=IMGSZ, classes=[BIRD_IDX], verbose=False
        )[0]
        bb = best_bird_box(r)

        if not bb:
            reasons["no_bird"] += 1
            if saved_no_bird < SAVE_NO_BIRD:
                d = DEBUG_DIR / "no_bird"
                d.mkdir(parents=True, exist_ok=True)
                img.save(d / f"{cls}_{src.stem}.jpg", quality=92)
                try:
                    arr = r.plot()
                    Image.fromarray(arr[..., ::-1]).save(d / f"{cls}_{src.stem}_det.jpg", quality=92)
                except Exception:
                    pass
                saved_no_bird += 1
            continue

        xyxy, score = bb
        w, h = img.size
        area = max(0.0, float((xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])))

        if score < CONF:
            reasons["low_conf"] += 1
            continue
        if area < (w * h * MIN_AREA_FRAC):
            reasons["too_small"] += 1
            continue

        xc, yc, bw, bh = pad_box_to_yolo_xywh(xyxy, w, h, PAD)
        line = f"{cls2id[cls]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n"
        kept_records.append((src, cls, line, xyxy))
        reasons["kept"] += 1

        if len(kept_records) % 200 == 0:
            bar.set_postfix(kept=reasons["kept"], no_bird=reasons["no_bird"])

    if not kept_records:
        raise RuntimeError("No images with detected birds found. Try lowering CONF or checking the detector weights.")

    # Split only on kept images, stratified by species
    kept_items = [(p, c) for (p, c, _line, _xyxy) in kept_records]
    train_items, val_items, test_items = stratified_split(kept_items, TRAIN_FRAC, VAL_FRAC, TEST_FRAC)

    # Lookup maps
    label_map = {p: line for (p, _c, line, _xyxy) in kept_records}
    box_map = {p: xyxy for (p, _c, _line, xyxy) in kept_records}

    def materialize(split: str, split_items: list[tuple[Path, str]]) -> None:
        # YOLO detect dataset
        y_img_out = OUT_YOLO / "images" / split
        y_lbl_out = OUT_YOLO / "labels" / split
        # Crops dataset in ImageNet format
        c_out = OUT_CROPS / split

        bar2 = tqdm(split_items, desc=f"Write {split} (yolo+crop)", unit="img")
        for src, cls in bar2:
            # keep <class>/<filename>.jpg
            rel = Path(cls) / (src.stem + ".jpg")

            dst_img = y_img_out / rel
            dst_lbl = (y_lbl_out / rel).with_suffix(".txt")
            safe_copy_as_jpg(src, dst_img)
            dst_lbl.parent.mkdir(parents=True, exist_ok=True)
            dst_lbl.write_text(label_map[src], encoding="utf-8")

            crop_dst = c_out / rel
            crop_dst.parent.mkdir(parents=True, exist_ok=True)

            # open once for cropping
            img = Image.open(src).convert("RGB")
            crop = crop_pad_pil(img, box_map[src], PAD)
            crop = maybe_resize_max_side(crop, CROP_MAX_SIDE)
            crop.save(crop_dst, quality=CROP_JPEG_QUALITY, subsampling=0)

    OUT_YOLO.mkdir(parents=True, exist_ok=True)
    OUT_CROPS.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        ensure_split_class_dirs(OUT_CROPS, split, names)


    materialize("train", train_items)
    materialize("val", val_items)
    materialize("test", test_items)

    write_yaml(names)

    # Stats
    print("\nDone.")
    print(f"- Input root: {IN_ROOT.resolve()}")
    print(f"- YOLO dataset: {OUT_YOLO.resolve()}")
    print(f"- Crops dataset: {OUT_CROPS.resolve()}")
    print(f"- YAML: {YAML_OUT.resolve()}")
    print("Stats:", dict(reasons))

    print("\nTrain YOLO detect (multi-class species) e.g.:")
    print(f"  yolo detect train model=yolo11s.pt data={YAML_OUT.as_posix()} imgsz={IMGSZ} epochs=50 device={DEV}")
    print("\nTrain classifier (ImageNet-style crops) e.g.:")
    print(f"  python your_mobilenet_script.py --data_root {OUT_CROPS.as_posix()}")


if __name__ == "__main__":
    main()
