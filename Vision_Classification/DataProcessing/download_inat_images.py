#!/usr/bin/env python3
"""
Download iNaturalist images for bird species classification.

This script reconstructs the image dataset from a metadata CSV file.
It reads observation metadata and downloads images from iNaturalist while
respecting API rate limits and maintaining license attribution.

Usage:
    python download_inat_images.py

Output:
    data/images/
    ├── species_name/
    │   ├── {observation_id}_{photo_id}.jpg
    │   └── ...
    └── download.log

Requirements:
    - requests
    - tqdm
    - pandas
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

# Configuration
METADATA_CSV = Path(__file__).parent / "metadata.csv"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "images"
LOG_FILE = OUTPUT_DIR / "download.log"
TIMEOUT = 30
SLEEP_BETWEEN_DOWNLOADS = 0.5
MAX_RETRIES = 3

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def setup_directories() -> None:
    """Create necessary directories."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def download_file(url: str, destination: Path, retries: int = MAX_RETRIES) -> bool:
    """Download file from URL with retry logic.
    
    Args:
        url: Source URL
        destination: Target file path
        retries: Number of retry attempts
        
    Returns:
        True if successful, False otherwise
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    if destination.exists():
        return True
    
    for attempt in range(retries):
        try:
            response = requests.get(url, stream=True, timeout=TIMEOUT)
            response.raise_for_status()
            
            # Write to temporary file first
            temp_file = destination.with_suffix(destination.suffix + ".tmp")
            with open(temp_file, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            # Move to final location
            temp_file.replace(destination)
            return True
            
        except requests.RequestException as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                logger.warning(f"Failed to download: {url} ({e})")
                return False
    
    return False


def process_metadata(metadata_csv: Path) -> list[dict]:
    """Load and validate metadata from CSV.
    
    Args:
        metadata_csv: Path to metadata CSV file
        
    Returns:
        List of record dictionaries
    """
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")
    
    records = []
    with open(metadata_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip records without valid URLs
            if not row.get("image_url") or row["image_url"] == "unknown":
                continue
            
            records.append(row)
    
    return records


def main():
    """Main download routine."""
    parser = argparse.ArgumentParser(description="Download iNaturalist images")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of images to download (for testing)",
    )
    parser.add_argument(
        "--species",
        type=str,
        default=None,
        help="Download only specific species (comma-separated)",
    )
    args = parser.parse_args()
    
    # Setup
    setup_directories()
    
    logger.info("=" * 70)
    logger.info("iNaturalist Image Dataset Download")
    logger.info("=" * 70)
    
    # Load metadata
    logger.info(f"\nLoading metadata from {METADATA_CSV}...")
    records = process_metadata(METADATA_CSV)
    logger.info(f"Found {len(records)} images with valid URLs")
    
    # Filter by species if specified
    if args.species:
        species_list = [s.strip() for s in args.species.split(",")]
        records = [
            r for r in records
            if r.get("inat_species_name") in species_list
        ]
        logger.info(f"Filtered to {len(records)} images for species: {args.species}")
    
    # Limit if specified
    if args.limit:
        records = records[:args.limit]
        logger.info(f"Limited to {len(records)} images")
    
    # Download images
    logger.info(f"\nDownloading images to {OUTPUT_DIR}...")
    
    success_count = 0
    skip_count = 0
    fail_count = 0
    
    progress = tqdm(records, desc="Downloading", unit="image")
    
    for record in progress:
        url = record["image_url"]
        species = record["inat_species_name"]
        obs_id = record["observation_id"]
        photo_id = record["photo_id"]
        
        # Build filename
        filename = f"{obs_id}_{photo_id}.jpg"
        filepath = OUTPUT_DIR / species / filename
        
        # Check if already exists
        if filepath.exists():
            skip_count += 1
            continue
        
        # Download
        if download_file(url, filepath):
            success_count += 1
        else:
            fail_count += 1
        
        # Rate limiting
        time.sleep(SLEEP_BETWEEN_DOWNLOADS)
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("Download Complete")
    logger.info("=" * 70)
    logger.info(f"Downloaded: {success_count}")
    logger.info(f"Skipped (already exists): {skip_count}")
    logger.info(f"Failed: {fail_count}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
