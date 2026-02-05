#!/usr/bin/env python3
"""
Extract bird crops from downloaded images using YOLOv11.

Processes images downloaded by download_inat_images.py and extracts bird
crops using YOLOv11 object detection. Only images with detected birds are saved.

Usage:
    python prepare_crops_with_yolov11.py

Input:
    data/images/{species}/*.jpg (from download_inat_images.py)

Output:
    data/images_cropped/{species}/*.jpg
    crop_processing.log
"""

from __future__ import annotations

import os
import csv
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
DEVICE = "mps" if torch.backends.mps.is_available() else ("0" if torch.cuda.is_available() else "cpu")

SOURCE_DIR = Path("data") / "images"
OUTPUT_DIR = Path("data") / "images_cropped"
LOG_FILE = OUTPUT_DIR / "crop_processing.log"

YOLO_MODEL = "yolo11s.pt"
BIRD_CLASS_IDX = 14
CONF_THRESHOLD = 0.1
IOU_THRESHOLD = 0.5
PADDING = 0.15
MIN_AREA_FRACTION = 0.005
MAX_SIDE = 1024
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def crop_pad(img: Image.Image, xyxy: np.ndarray, pad: float) -> Image.Image:
    """Crop image with padding around bounding box."""
    w, h = img.size
    x1, y1, x2, y2 = xyxy.astype(float)
    pw, ph = (x2 - x1) * pad, (y2 - y1) * pad
    x1 = int(max(0, x1 - pw))
    y1 = int(max(0, y1 - ph))
    x2 = int(min(w, x2 + pw))
    y2 = int(min(h, y2 + ph))
    return img.crop((x1, y1, x2, y2))


def resize_if_needed(img: Image.Image, max_side: int) -> Image.Image:
    """Resize image if any dimension exceeds max_side."""
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale = max_side / m
    return img.resize((int(w * scale), int(h * scale)))


def extract_bird_box(detection_result) -> tuple[np.ndarray, float] | None:
    """Extract highest confidence bird detection box."""
    boxes = detection_result.boxes
    if boxes is None or len(boxes) == 0:
        return None
    
    classes = boxes.cls.detach().cpu().numpy().astype(int)
    confidences = boxes.conf.detach().cpu().numpy().astype(float)
    
    bird_indices = np.where(classes == BIRD_CLASS_IDX)[0]
    if bird_indices.size == 0:
        return None
    
    best_idx = bird_indices[np.argmax(confidences[bird_indices])]
    return boxes.xyxy[best_idx].detach().cpu().numpy(), float(confidences[best_idx])


def main():
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Source directory not found: {SOURCE_DIR}")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("Extract Bird Crops with YOLOv11")
    logger.info("=" * 70)
    
    logger.info(f"\nLoading YOLOv11 model ({YOLO_MODEL})...")
    model = YOLO(YOLO_MODEL)
    
    species_images = {}
    for img_path in SOURCE_DIR.rglob("*.jpg"):
        if img_path.suffix.lower() not in ALLOWED_EXTENSIONS:
            continue
        species = img_path.parent.name
        if species not in species_images:
            species_images[species] = []
        species_images[species].append(img_path)
    
    total_images = sum(len(v) for v in species_images.values())
    logger.info(f"\nFound {total_images} images ({len(species_images)} species)")
    
    logger.info(f"\nProcessing images...")
    
    success = skip = fail = 0
    progress = tqdm(total=total_images, desc="Processing", unit="image")
    
    for species, images in sorted(species_images.items()):
        output_species_dir = OUTPUT_DIR / species
        output_species_dir.mkdir(parents=True, exist_ok=True)
        
        for img_path in images:
            output_path = output_species_dir / img_path.name
            
            if output_path.exists():
                skip += 1
                progress.update(1)
                continue
            
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                fail += 1
                progress.update(1)
                continue
            
            try:
                result = model.predict(
                    img,
                    device=DEVICE,
                    conf=CONF_THRESHOLD,
                    iou=IOU_THRESHOLD,
                    verbose=False
                )[0]
            except Exception:
                fail += 1
                progress.update(1)
                continue
            
            box_data = extract_bird_box(result)
            if not box_data:
                fail += 1
                progress.update(1)
                continue
            
            xyxy, conf = box_data
            w, h = img.size
            area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
            
            if conf < CONF_THRESHOLD or area < (w * h * MIN_AREA_FRACTION):
                fail += 1
                progress.update(1)
                continue
            
            try:
                crop = resize_if_needed(crop_pad(img, xyxy, PADDING), MAX_SIDE)
                crop.save(output_path, quality=92, subsampling=0)
                success += 1
            except Exception:
                fail += 1
            
            progress.update(1)
    
    progress.close()
    
    logger.info("\n" + "=" * 70)
    logger.info("Processing Complete")
    logger.info("=" * 70)
    logger.info(f"Successful: {success}")
    logger.info(f"Skipped (existing): {skip}")
    logger.info(f"Failed: {fail}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
