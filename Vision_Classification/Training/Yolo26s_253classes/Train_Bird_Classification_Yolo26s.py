"""
YOLO Training Pipeline for Weakly Supervised Datasets.

Maintains compatibility with output structures from automated conversion scripts.
Supports recursive label searching and multi-split evaluation.

"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import yaml
from ultralytics import YOLO


# -----------------------------
# CONFIG (anpassen)
# -----------------------------
@dataclass
class Config:
    """Training configuration and hyperparameters."""
    DET_DATA_YAML: str = "iNat_species_yolo_det.yaml"
    DEVICE: str = ""

    # Model parameters
    DET_MODEL: str = "yolo26s.pt"
    DET_EPOCHS: int = 150
    DET_PATIENCE: int = 30
    DET_IMGSZ: int = 640
    DET_BATCH: int = 32
    DET_WORKERS: int = 8

    # Output management
    DET_RUNS_DIR: str = "runs/yolo26"
    DET_RUN_NAME: str = "yolo_26s"
    DET_VAL_SPLIT: str = "val"
    DET_PLOTS: bool = True
    DET_SAVE_JSON: bool = False 
    SEED: int = 7

    # Optimizer settings
    DET_OPTIMIZER: str = "auto"
    DET_LR0: float = 0.002
    DET_LRF: float = 0.01
    DET_WEIGHT_DECAY: float = 0.001
    DET_MOMENTUM: float = 0.937

    # Augmentations
    DET_CLOSE_MOSAIC: int = 10
    DET_TRANSLATE: float = 0.05
    DET_SCALE: float = 0.3
    DET_FLIPLR: float = 0.5
    DET_HSV_H: float = 0.01
    DET_HSV_S: float = 0.5
    DET_HSV_V: float = 0.3

    # Loss scaling
    DET_BOX_WEIGHT: float = 6.0
    DET_CLS_WEIGHT: float = 1.5
    DET_DFL_WEIGHT: float = 1.5
    DET_LABEL_SMOOTHING: float = 0.02

    # Inference thresholds
    DET_CONF_THRESHOLD: float = 0.001
    DET_IOU_THRESHOLD: float = 0.7


CFG = Config()


def get_device(cfg_device: str) -> str:
    """Selects the best available hardware accelerator."""
    if cfg_device:
        return cfg_device
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def read_yaml(path: str | Path) -> Dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"YAML nicht gefunden: {p.resolve()}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def print_det_metrics(metrics, title: str) -> Dict[str, float]:
    """Extracts and logs primary detection performance indicators."""
    box = metrics.box
    mp = getattr(box, "mp", None)
    mr = getattr(box, "mr", None)

    map50 = float(getattr(box, "map50", float("nan")))
    map5095 = float(getattr(box, "map", float("nan")))
    map75 = float(getattr(box, "map75", float("nan")))

    f1 = float("nan")
    if mp is not None and mr is not None and (mp + mr) > 0:
        f1 = float(2 * mp * mr / (mp + mr))

    out = {
        "precision_mean": float(mp) if mp is not None else float("nan"),
        "recall_mean": float(mr) if mr is not None else float("nan"),
        "f1_mean": f1,
        "mAP50": map50,
        "mAP75": map75,
        "mAP50-95": map5095,
    }

    print("\n" + "=" * 70)
    print(title)
    print("-" * 70)
    for k, v in out.items():
        print(f"{k:>14}: {v:.6f}" if isinstance(v, float) else f"{k:>14}: {v}")

    maps = getattr(box, "maps", None)
    if maps is not None:
        print("\nPer-class mAP50-95:")
        for i, ap in enumerate(list(maps)):
            print(f"  class {i:>3}: {float(ap):.6f}")
    print("=" * 70 + "\n")

    return out


def verify_det_dataset_yaml(det_yaml_path: Path) -> Tuple[Path, Dict]:
    """Ensures the dataset follows the expected YOLO recursive directory structure."""
    y = read_yaml(det_yaml_path)
    if "path" not in y:
        raise ValueError("Det-YAML hat kein 'path:' Feld (erwartet vom Converter).")
    base = Path(y["path"])

    # check split paths exist
    for k in ("train", "val", "test"):
        if k in y:
            p = base / str(y[k])
            if not p.exists():
                raise FileNotFoundError(f"Split-Pfad fehlt: {k} -> {p.resolve()}")

    # check label dirs exist + contain labels (rekursiv)
    for k in ("train", "val", "test"):
        if k in y:
            img_dir = base / str(y[k])
            lbl_dir = base / "labels" / img_dir.name
            if not lbl_dir.exists():
                print(f"[WARN] Label-Verzeichnis nicht gefunden (erwartet): {lbl_dir.resolve()}")
            else:
                n_lbl = len(list(lbl_dir.rglob("*.txt")))
                if n_lbl == 0:
                    print(f"[WuARN] Keine Label-Dateien unter: {lbl_dir.resolve()} (auch nicht rekursiv)")

    if "names" not in y:
        print("[WARN] Det-YAML hat kein 'names'. Training klappt oft trotzdem, aber Klassennamen fehlen.")
    return base, y


def train_detection(cfg: Config, device: str) -> Tuple[Path, Dict[str, float]]:
    """Executes the training and evaluation loop."""
    det_yaml = Path(cfg.DET_DATA_YAML)
    base, y = verify_det_dataset_yaml(det_yaml)

    # Label Quality Check
    print("\n" + "=" * 70)
    print("WEAK SUPERVISION - Label Quality Check")
    print("=" * 70)

    train_lbl_root = base / "labels" / "train"
    train_labels = list(train_lbl_root.rglob("*.txt"))
    print(f"Training labels gefunden (rekursiv): {len(train_labels)}")

    if len(train_labels) == 0:
        print(f"❌ Keine Labeldateien gefunden unter: {train_lbl_root.resolve()}")
        print("   -> Prüfe Ordnerstruktur oder YAML 'path:'")
        print("=" * 70 + "\n")
    else:
        sample = train_labels[:100] if len(train_labels) >= 100 else train_labels

        total_boxes = 0
        empty_files = 0
        bad_lines = 0

        for lbl in sample:
            lines = lbl.read_text(encoding="utf-8").strip().splitlines()
            if not lines:
                empty_files += 1
                continue
            for ln in lines:
                ln = ln.strip()
                if not ln:
                    continue
                parts = ln.split()
                if len(parts) != 5:
                    bad_lines += 1
                    continue
                total_boxes += 1

        denom = max(1, len(sample))
        avg_boxes = total_boxes / denom
        print(f"Sample size: {denom}")
        print(f"Durchschnitt Boxes/Labeldatei (Sample): {avg_boxes:.2f}")
        print(f"Leere Label-Dateien: {empty_files}/{denom}")
        print(f"Bad label lines (!=5 Spalten): {bad_lines}")

        if avg_boxes < 0.5:
            print("⚠️  WARNING: Sehr wenige Boxen pro Datei im Sample. Weak Labels evtl. sehr dünn/noisy?")
        print("=" * 70 + "\n")

    # Initialize Model
    model = YOLO(cfg.DET_MODEL)

    model.train(
        data=str(det_yaml),
        epochs=cfg.DET_EPOCHS,
        imgsz=cfg.DET_IMGSZ,
        batch=cfg.DET_BATCH,
        device=device,
        workers=cfg.DET_WORKERS,
        project=cfg.DET_RUNS_DIR,
        name=cfg.DET_RUN_NAME,
        exist_ok=True,
        cache="ram",
        amp=True,
        patience=cfg.DET_PATIENCE,
        val=True,
        plots=cfg.DET_PLOTS,
        save=True,
        save_period=15,
        verbose=True,
        optimizer=cfg.DET_OPTIMIZER,
        lr0=cfg.DET_LR0,
        lrf=cfg.DET_LRF,
        momentum=cfg.DET_MOMENTUM,
        weight_decay=cfg.DET_WEIGHT_DECAY,
        warmup_epochs=2.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        mosaic=0.2,
        mixup=0.0,
        copy_paste=0.0,
        close_mosaic=cfg.DET_CLOSE_MOSAIC,

        translate=cfg.DET_TRANSLATE,
        scale=cfg.DET_SCALE,
        fliplr=cfg.DET_FLIPLR,

        hsv_h=cfg.DET_HSV_H,
        hsv_s=cfg.DET_HSV_S,
        hsv_v=cfg.DET_HSV_V,

        # Loss
        box=cfg.DET_BOX_WEIGHT,
        cls=cfg.DET_CLS_WEIGHT,
        dfl=cfg.DET_DFL_WEIGHT,
        label_smoothing=cfg.DET_LABEL_SMOOTHING,

        rect=False,
        cos_lr=True,
        dropout=0.0,
    )

    if not hasattr(model, "trainer") or model.trainer is None:
        raise RuntimeError("Ultralytics trainer nicht vorhanden")

    save_dir = Path(model.trainer.save_dir)

    results_csv = save_dir / "results.csv"
    if results_csv.exists():
        try:
            import pandas as pd
            df = pd.read_csv(results_csv)
            print("\n" + "=" * 70)
            print("TRAINING SUMMARY")
            print("=" * 70)
            if "metrics/mAP50(B)" in df.columns:
                print(f"Best mAP50: {df['metrics/mAP50(B)'].max():.4f}")
            if "metrics/mAP50-95(B)" in df.columns:
                print(f"Best mAP50-95: {df['metrics/mAP50-95(B)'].max():.4f}")
            if "train/box_loss" in df.columns:
                print(f"Final box_loss: {df['train/box_loss'].iloc[-1]:.4f}")
            if "train/cls_loss" in df.columns:
                print(f"Final cls_loss: {df['train/cls_loss'].iloc[-1]:.4f}")
            print("=" * 70 + "\n")
        except Exception as e:
            print(f"[WARN] Konnte results.csv nicht auswerten: {e}")

    best_pt = save_dir / "weights" / "best.pt"
    if not best_pt.exists():
        last_pt = save_dir / "weights" / "last.pt"
        if last_pt.exists():
            print("[WARN] best.pt fehlt, nutze last.pt")
            best_pt = last_pt
        else:
            raise FileNotFoundError("Keine weights gefunden")

    best_model = YOLO(str(best_pt))
    
    # Validation
    metrics = best_model.val(
        data=str(det_yaml),
        split=cfg.DET_VAL_SPLIT,
        device=device,
        plots=True,
        save_json=cfg.DET_SAVE_JSON,
        max_det=100,
        imgsz=cfg.DET_IMGSZ,
        conf=cfg.DET_CONF_THRESHOLD,
        iou=cfg.DET_IOU_THRESHOLD,
        verbose=True,
    )

    summary = print_det_metrics(metrics, title=f"YOLO26 Detection Metrics ({cfg.DET_VAL_SPLIT})")

    # Train split (overfit check)
    metrics_train = best_model.val(
        data=str(det_yaml),
        split="train",
        device=device,
        plots=False,
        save_json=False,
        max_det=300,
        imgsz=cfg.DET_IMGSZ,
        conf=0.001,
        iou=0.7,
        verbose=False,
    )
    summary_train = print_det_metrics(metrics_train, title="YOLO26 Detection Metrics (train)")


    summary_path = save_dir / f"metrics_{cfg.DET_VAL_SPLIT}.summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    train_summary_path = save_dir / "metrics_train.summary.json"
    with train_summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_train, f, indent=2, ensure_ascii=False)
    print(f"[OK] Train metrics summary: {train_summary_path.resolve()}")


    print(f"\n[OK] Detection Dataset base: {base.resolve()}")
    print(f"[OK] Best weights: {best_pt.resolve()}")
    print(f"[OK] Metrics summary: {summary_path.resolve()}")
    print(f"[OK] Training curves: {save_dir / 'results.png'}")

    return best_pt, summary


def main() -> None:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    set_seed(CFG.SEED)

    device = get_device(CFG.DEVICE)
    print(f"[INFO] Using device: {device}")

    train_detection(CFG, device)


if __name__ == "__main__":
    main()