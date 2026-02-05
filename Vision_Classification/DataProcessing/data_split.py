#!/usr/bin/env python3
"""
Stratified train/val/test split for cropped bird images.

Performs 60/20/20 stratified split per species on cropped images from
prepare_crops_with_yolov11.py. Creates clean directory structure for training.

Usage:
    python data_split.py

Output:
    data/
    ├── train/{species}/*.jpg
    ├── val/{species}/*.jpg
    ├── test/{species}/*.jpg
    ├── metadata.csv
    └── split_stats.json
"""

from __future__ import annotations

import json
import csv
import shutil
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Configuration
SOURCE_DIR = Path("data") / "images_cropped"
OUTPUT_DIR = Path("data")
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2
RANDOM_STATE = 42


def collect_images():
    """Collect all images from source directory, organized by species."""
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Source directory not found: {SOURCE_DIR}")
    
    species_images = defaultdict(list)
    for img_path in SOURCE_DIR.rglob("*.jpg"):
        species = img_path.parent.name
        species_images[species].append(img_path)
    
    total = sum(len(v) for v in species_images.values())
    print(f"\n📁 Collected {total} images ({len(species_images)} species)")
    
    return species_images


def stratified_split(species_images):
    """Perform 60/20/20 stratified split per species."""
    train_data = defaultdict(list)
    val_data = defaultdict(list)
    test_data = defaultdict(list)
    
    print(f"\n🔀 Stratified split (60/20/20)...")
    
    for species, images in sorted(species_images.items()):
        n = len(images)
        
        if n == 0:
            continue
        elif n == 1:
            train_data[species] = images
        elif n == 2:
            train, val = train_test_split(images, test_size=0.5, random_state=RANDOM_STATE)
            train_data[species] = train
            val_data[species] = val
        elif n == 3:
            train_val, _ = train_test_split(images, test_size=1/3, random_state=RANDOM_STATE)
            train, val = train_test_split(train_val, test_size=0.5, random_state=RANDOM_STATE)
            train_data[species] = train
            val_data[species] = val
        else:
            train, temp = train_test_split(images, test_size=0.4, random_state=RANDOM_STATE)
            val, test = train_test_split(temp, test_size=0.5, random_state=RANDOM_STATE + 1)
            train_data[species] = train
            val_data[species] = val
            test_data[species] = test
    
    n_train = sum(len(v) for v in train_data.values())
    n_val = sum(len(v) for v in val_data.values())
    n_test = sum(len(v) for v in test_data.values())
    n_total = n_train + n_val + n_test
    
    print(f"   Train: {n_train} ({100*n_train/n_total:.1f}%)")
    print(f"   Val:   {n_val} ({100*n_val/n_total:.1f}%)")
    print(f"   Test:  {n_test} ({100*n_test/n_total:.1f}%)")
    
    return train_data, val_data, test_data


def create_structure(train_data, val_data, test_data):
    """Create directory structure and copy files."""
    print(f"\n📂 Creating structure...")
    
    for split_name in ["train", "val", "test"]:
        split_dir = OUTPUT_DIR / split_name
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)
    
    splits = [
        ("train", train_data),
        ("val", val_data),
        ("test", test_data),
    ]
    
    for split_name, split_data in splits:
        print(f"\n   Copying {split_name}...")
        for species, images in tqdm(split_data.items(), desc=split_name, leave=False):
            if not images:
                continue
            species_dir = OUTPUT_DIR / split_name / species
            species_dir.mkdir(parents=True, exist_ok=True)
            for img_path in images:
                shutil.copy2(img_path, species_dir / img_path.name)


def generate_metadata(train_data, val_data, test_data):
    """Generate metadata CSV and statistics."""
    print(f"\n📊 Generating metadata...")
    
    metadata_file = OUTPUT_DIR / "metadata_split.csv"
    with open(metadata_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["species", "split", "image_file", "image_path"])
        
        for species, images in train_data.items():
            for img in images:
                writer.writerow([species, "train", img.name, f"train/{species}/{img.name}"])
        
        for species, images in val_data.items():
            for img in images:
                writer.writerow([species, "val", img.name, f"val/{species}/{img.name}"])
        
        for species, images in test_data.items():
            for img in images:
                writer.writerow([species, "test", img.name, f"test/{species}/{img.name}"])
    
    stats = {
        "total": sum(len(v) for d in [train_data, val_data, test_data] for v in d.values()),
        "train": sum(len(v) for v in train_data.values()),
        "val": sum(len(v) for v in val_data.values()),
        "test": sum(len(v) for v in test_data.values()),
        "species": len(set(list(train_data.keys()) + list(val_data.keys()) + list(test_data.keys()))),
    }
    
    stats_file = OUTPUT_DIR / "split_stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"   ✓ metadata.csv")
    print(f"   ✓ split_stats.json")
    
    return stats


def main():
    print("=" * 70)
    print("Stratified Train/Val/Test Split")
    print("=" * 70)
    
    species_images = collect_images()
    train_data, val_data, test_data = stratified_split(species_images)
    create_structure(train_data, val_data, test_data)
    stats = generate_metadata(train_data, val_data, test_data)
    
    print(f"\n{'='*70}")
    print(f"✓ Split complete")
    print(f"{'='*70}")
    print(f"\nOutput: {OUTPUT_DIR}")
    print(f"Total: {stats['total']} images ({stats['species']} species)")


if __name__ == "__main__":
    main()
