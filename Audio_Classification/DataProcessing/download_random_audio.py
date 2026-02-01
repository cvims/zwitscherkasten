"""Download random bird audio from Xeno-Canto.

This script selects a random subset of species from the CSV
`DataProcessing/Vogel_Vergleich_Liste_Birdset_vs_XenoCanto.csv`, queries the
Xeno-Canto API for high-quality recordings within a duration limit, saves
per-species metadata into `sampled_metadata/`, and downloads audio files into
`audio_data/<species>/`.

Usage:
    - Set your Xeno Canto API key in `API_KEY`.
    - Run: `python DataProcessing/download_random_audio.py`

Configuration (see top of file): API_KEY, CSV_FILE, MAX_DURATION_SEC,
NUM_RANDOM_SPECIES, FILES_PER_SPECIES, and worker/timeouts for concurrency.

Note: The script uses thread pools for searching and downloading and is
intended for small-scale sampling and data collection for audio experiments.
"""

import json
import logging
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd
import requests

# ==============================================================================
# ⚙️ CONFIGURATION
# ==============================================================================

API_KEY = "YOUR_XENO_CANTO_API_KEY_HERE" 
CSV_FILE = Path("DataProcessing/Vogel_Vergleich_Liste_Birdset_vs_XenoCanto.csv")

MAX_DURATION_SEC = 60           # Max recording length
TOP_N_CLASSES = 256             # Top species in CSV
NUM_RANDOM_SPECIES = 10         # Number of species to randomly select
FILES_PER_SPECIES = 3           # Audio files per species

SEARCH_WORKERS = 16
DOWNLOAD_WORKERS = 16
REQUEST_TIMEOUT = 15
DOWNLOAD_TIMEOUT = 60
API_DELAY = 0.2

METADATA_DIR = Path("sampled_metadata")
AUDIO_DIR = Path("audio_data")

# ==============================================================================
# 📝 LOGGING SETUP
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ==============================================================================
# 🔧 UTILITIES
# ==============================================================================


def load_species(csv_path: Path, top_n: int, random_n: int) -> List[str]:
    """Load top species from CSV and select a random subset."""
    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        return []

    df = pd.read_csv(csv_path, delimiter=";")
    if "Wissenschaftlicher Name" not in df.columns:
        logger.error("Column 'Wissenschaftlicher Name' not found in CSV")
        return []

    species_list = df.iloc[:top_n]["Wissenschaftlicher Name"].dropna().str.strip().tolist()

    if len(species_list) > random_n:
        selected = random.sample(species_list, random_n)
    else:
        selected = species_list
        logger.warning(f"Only {len(species_list)} species available, requested {random_n}")

    logger.info(f"Selected {len(selected)} species from top {top_n}")
    return selected


def parse_duration(duration: str) -> float:
    """Convert a duration string (MM:SS or seconds) to float seconds."""
    try:
        if ":" in duration:
            mins, secs = duration.split(":")
            return int(mins) * 60 + float(secs)
        return float(duration)
    except Exception:
        return 0.0


def clean_recording(rec: Dict) -> Dict:
    """Extract essential recording metadata."""
    return {
        "id": rec["id"],
        "quality": rec.get("q"),
        "length": rec.get("length"),
        "file_url": rec["file"],
    }


# ==============================================================================
# 🔍 METADATA SEARCH
# ==============================================================================


def search_recordings(species: str, max_files: int) -> Tuple[str, List[Dict]]:
    """Search recordings for a species, filtered by quality and duration."""
    query = " ".join([f"gen:{species.split()[0]}", f"sp:{species.split()[1]}"][:2])
    base_url = "https://xeno-canto.org/api/3/recordings"

    collected: List[Dict] = []
    page = 1
    max_pages = 20

    while page <= max_pages and len(collected) < max_files:
        try:
            resp = requests.get(base_url, params={"query": query, "key": API_KEY, "page": page}, timeout=REQUEST_TIMEOUT)
            if resp.status_code != 200:
                logger.warning(f"[{species}] HTTP {resp.status_code} page {page}")
                break

            data = resp.json()
            recordings = data.get("recordings", [])
            if not recordings:
                break

            for rec in recordings:
                if len(collected) >= max_files:
                    break
                if rec.get("q") not in {"A", "B", "C"}:
                    continue
                if parse_duration(rec.get("length", "0")) > MAX_DURATION_SEC:
                    continue
                collected.append(clean_recording(rec))

            if len(collected) >= max_files:
                break

            total_pages = int(data.get("numPages", 1))
            if page >= total_pages:
                break

            page += 1
            time.sleep(API_DELAY)
        except Exception as e:
            logger.error(f"[{species}] Error on page {page}: {e}")
            break

    # Sort by quality A > B > C
    quality_order = {"A": 0, "B": 1, "C": 2}
    collected.sort(key=lambda x: quality_order.get(x.get("quality", "C"), 3))
    return species, collected[:max_files]


def save_metadata(species: str, recordings: List[Dict]):
    """Save species recordings metadata to JSON."""
    if not recordings:
        logger.warning(f"No recordings to save for {species}")
        return
    METADATA_DIR.mkdir(exist_ok=True)
    filepath = METADATA_DIR / f"{species.replace(' ', '_')}_metadata.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(recordings, f, indent=2, ensure_ascii=False)


# ==============================================================================
# ⬇️ AUDIO DOWNLOAD
# ==============================================================================


def download_audio(rec: Dict, dest_dir: Path) -> Tuple[bool, int]:
    """Download a single recording."""
    try:
        with requests.get(rec["file_url"], stream=True, timeout=DOWNLOAD_TIMEOUT) as r:
            if r.status_code != 200:
                logger.warning(f"Failed to download {rec['id']}: HTTP {r.status_code}")
                return False, 0

            dest_dir.mkdir(parents=True, exist_ok=True)
            filepath = dest_dir / f"{rec['id']}_q{rec['quality']}.mp3"
            size = 0
            with open(filepath, "wb") as f:
                for chunk in r.iter_content(8192):
                    if chunk:
                        f.write(chunk)
                        size += len(chunk)
            return True, size
    except Exception as e:
        logger.error(f"Download error {rec['id']}: {e}")
        return False, 0


# ==============================================================================
# 🚀 MAIN
# ==============================================================================


def main():
    print("=" * 70)
    print("🎵 XENO-CANTO DOWNLOADER")
    print(f"Filter: ≤ {MAX_DURATION_SEC}s | Files per species: {FILES_PER_SPECIES}")
    print(f"Random selection: {NUM_RANDOM_SPECIES} from top {TOP_N_CLASSES}")
    print("=" * 70)

    random.seed(42)

    species_list = load_species(CSV_FILE, TOP_N_CLASSES, NUM_RANDOM_SPECIES)
    if not species_list:
        logger.error("No species loaded! Exiting.")
        sys.exit(1)

    # ── PHASE 1: Metadata ──
    print("\n📋 Collecting metadata...")
    METADATA_DIR.mkdir(exist_ok=True)

    with ThreadPoolExecutor(max_workers=SEARCH_WORKERS) as executor:
        futures = {executor.submit(search_recordings, sp, FILES_PER_SPECIES): sp for sp in species_list}
        for future in as_completed(futures):
            sp, recs = future.result()
            save_metadata(sp, recs)
            print(f"Saved metadata for {sp} ({len(recs)} recordings)")

    # ── PHASE 2: Downloads ──
    print("\n⬇️ Downloading audio files...")
    download_jobs: List[Tuple[Dict, Path]] = []
    for metadata_file in METADATA_DIR.glob("*.json"):
        sp_name = metadata_file.stem.replace("_metadata", "").replace("_", " ")
        dest_dir = AUDIO_DIR / sp_name
        with open(metadata_file, "r", encoding="utf-8") as f:
            recs = json.load(f)
            for rec in recs:
                if not (dest_dir / f"{rec['id']}_q{rec['quality']}.mp3").exists():
                    download_jobs.append((rec, dest_dir))

    total_jobs = len(download_jobs)
    logger.info(f"{total_jobs} files to download")

    downloaded = 0
    total_bytes = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
        futures = [executor.submit(download_audio, rec, path) for rec, path in download_jobs]
        for i, future in enumerate(as_completed(futures), 1):
            success, size = future.result()
            if success:
                downloaded += 1
                total_bytes += size
            if i % 5 == 0 or i == total_jobs:
                elapsed = time.time() - start_time
                speed = (total_bytes / (1024 * 1024)) / elapsed if elapsed > 0 else 0
                print(f"Downloaded {downloaded}/{total_jobs} | {total_bytes/1e6:.1f} MB | {speed:.2f} MB/s", end="\r")

    print(f"\n✅ Download complete: {downloaded}/{total_jobs} files, {total_bytes/1e6:.1f} MB")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1)
