"""Download and manage bird audio recordings from Xeno‑Canto.

This module searches Xeno‑Canto for recordings of species listed in
`Vogel_Vergleich_Liste_Birdset_vs_XenoCanto.csv`, filters by quality and
maximum duration, writes per-species metadata JSON files, and downloads audio
files into `audio_data/<species>/`.

Configuration (see constants near the top of the file): set `API_KEY`,
`MAX_DURATION_SECONDS`, `START_INDEX/END_INDEX`, and download mode options.

Usage:
    - Set `API_KEY` to your Xeno‑canto API key.
    - Run: `python DataProcessing/download_audio_data.py`

Note: The script uses thread pools for concurrent searching and downloading
and is intended for batch data collection / sampling for audio experiments.
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Xeno‑canto API key.
API_KEY: str = "your_api_key_here" # Replace with your actual API key.

# CSV file with a column containing scientific bird names.
CSV_FILENAME: str = "Vogel_Vergleich_Liste_Birdset_vs_XenoCanto.csv"

# Smart filter: ignore recordings longer than this (in seconds).
# This saves download time and disk space.
MAX_DURATION_SECONDS: float = 60.0

# Range of species indices in the CSV to process.
# For example: first 64 species -> START_INDEX=0, END_INDEX=64 (end is exclusive).
START_INDEX: int = 0
END_INDEX: int = 128

# Download mode:
#   "UNLIMITED": get all matching recordings (subject to quality + duration filters)
#   "LIMITED":   get up to LIMIT_PER_CATEGORY per quality class (A/B/C)
DOWNLOAD_MODE: str = "UNLIMITED"
LIMIT_PER_CATEGORY: int = 30

# Thread pool sizes.
SEARCH_WORKERS: int = 16
DOWNLOAD_WORKERS: int = 16

# Xeno‑canto API endpoint.
XC_BASE_URL: str = "https://xeno-canto.org/api/3/recordings"

# ==============================================================================
# UTILS
# ==============================================================================


def log(msg: str) -> None:
    """Print a timestamped log message."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def load_species_from_csv(
    csv_path: str,
    start: int = 0,
    end: Optional[int] = None,
    column_name: str = "Wissenschaftlicher Name",
) -> List[str]:
    """
    Load a list of scientific species names from a CSV file.

    Parameters
    ----------
    csv_path:
        Path to the CSV file.
    start:
        Start index (inclusive) in the CSV.
    end:
        End index (exclusive) in the CSV. If None, use all rows from `start`.
    column_name:
        Name of the column that contains the scientific names.

    Returns
    -------
    list of str
        Cleaned list of non-empty scientific names.
    """
    try:
        df = pd.read_csv(csv_path, delimiter=";")
    except Exception as exc:  # noqa: BLE001
        log(f"Failed to read CSV '{csv_path}': {exc}")
        return []

    if column_name not in df.columns:
        log(f"Column '{column_name}' not found in CSV.")
        return []

    start = max(start, 0)
    slice_df = df.iloc[start:end] if end is not None else df.iloc[start:]
    species_list = slice_df[column_name].tolist()

    cleaned = [
        str(name).strip()
        for name in species_list
        if isinstance(name, str) and str(name).strip()
    ]
    return cleaned


def parse_duration(duration_str: object) -> float:
    """
    Parse a duration string from Xeno‑canto into seconds.

    Examples
    --------
    "0:32" -> 32.0
    "1:05" -> 65.0
    "15"   -> 15.0

    Any invalid value falls back to 0.0.
    """
    try:
        s = str(duration_str)
        if ":" in s:
            minutes, seconds = s.split(":", maxsplit=1)
            return int(minutes) * 60 + float(seconds)
        return float(s)
    except (ValueError, TypeError):
        return 0.0


def clean_rec_data(rec: Dict) -> Dict:
    """
    Reduce a raw recording dict from the API to the fields needed later.

    This keeps the metadata JSON small and makes downstream code simpler.
    """
    return {
        "id": rec.get("id"),
        "quality": rec.get("q"),
        "length": rec.get("length"),
        "file_url": rec.get("file"),
    }


def build_query_from_species(species_name: str) -> str:
    """
    Build a Xeno‑canto query string from a scientific species name.

    Xeno‑canto query syntax:
    - "gen:<genus> sp:<species>" if both genus and species are present
    - "gen:<genus>" as fallback if only genus is available
    """
    parts = species_name.split()
    if len(parts) >= 2:
        return f"gen:{parts[0]} sp:{parts[1]}"
    return f"gen:{parts[0]}"


def fetch_recording_page(
    query: str,
    page: int,
    timeout: int = 15,
) -> Optional[Dict]:
    """
    Fetch a single page of Xeno‑canto recordings for a given query.

    Returns the JSON payload as dict, or None if the request failed.
    """
    try:
        response = requests.get(
            XC_BASE_URL,
            params={"query": query, "key": API_KEY, "page": page},
            timeout=timeout,
        )
        if response.status_code != 200:
            log(f"API error (status {response.status_code}) for '{query}', page {page}")
            return None
        return response.json()
    except requests.RequestException as exc:  # noqa: BLE001
        log(f"HTTP error for '{query}', page {page}: {exc}")
        return None


# ==============================================================================
# SEARCH / METADATA
# ==============================================================================


def search_unlimited(species_name: str) -> Tuple[str, List[Dict]]:
    """
    Fetch all recordings for one species (unlimited mode).

    Filters:
    - quality in {A, B, C}
    - duration <= MAX_DURATION_SECONDS
    """
    query = build_query_from_species(species_name)
    log(f"[{species_name}] Searching (unlimited mode)...")

    collected: List[Dict] = []
    page = 1

    while True:
        data = fetch_recording_page(query, page, timeout=15)
        if not data or not data.get("recordings"):
            break

        for rec in data["recordings"]:
            if rec.get("q") not in {"A", "B", "C"}:
                continue
            if parse_duration(rec.get("length", "0")) > MAX_DURATION_SECONDS:
                continue
            collected.append(clean_rec_data(rec))

        num_pages = int(data.get("numPages", 1))
        if page >= num_pages:
            break

        page += 1
        time.sleep(0.2)  # Be nice to the API.

    return species_name, collected


def search_limited(species_name: str, limit: int) -> Tuple[str, List[Dict]]:
    """
    Fetch up to `limit` recordings per quality class (A/B/C) for one species.

    This is useful if you want a more balanced dataset across quality levels.
    """
    query = build_query_from_species(species_name)
    log(f"[{species_name}] Searching (limit {limit} per quality class)...")

    collected: Dict[str, List[Dict]] = {"A": [], "B": [], "C": []}
    page = 1
    max_pages = 50  # Safety cap in case the API behaves unexpectedly.

    while page <= max_pages:
        data = fetch_recording_page(query, page, timeout=10)
        if not data or not data.get("recordings"):
            break

        for rec in data["recordings"]:
            quality = rec.get("q", "E")
            if quality not in collected:
                continue
            if parse_duration(rec.get("length", "0")) > MAX_DURATION_SECONDS:
                continue
            if len(collected[quality]) < limit:
                collected[quality].append(clean_rec_data(rec))

        # Stop early if all quality classes reached their limit.
        if all(len(collected[q]) >= limit for q in collected):
            break

        page += 1
        time.sleep(0.2)

    merged: List[Dict] = collected["A"] + collected["B"] + collected["C"]
    return species_name, merged


def save_metadata(species_name: str, recordings: Iterable[Dict]) -> None:
    """
    Save metadata for one species as JSON.

    Each species gets its own file:
        metadata/<Genus_species>_metadata.json
    """
    recordings = list(recordings)
    if not recordings:
        return

    metadata_dir = Path("metadata")
    metadata_dir.mkdir(exist_ok=True)

    safe_name = species_name.replace(" ", "_")
    filename = metadata_dir / f"{safe_name}_metadata.json"

    try:
        with filename.open("w", encoding="utf-8") as f:
            json.dump(recordings, f, indent=2)
    except OSError as exc:  # noqa: BLE001
        log(f"Failed to write metadata for '{species_name}': {exc}")


# ==============================================================================
# DOWNLOAD
# ==============================================================================


def download_single_file(
    rec: Dict,
    species_dir: Path,
    session: Optional[requests.Session] = None,
) -> Tuple[bool, int]:
    """
    Download one audio file and return (success, size_in_bytes).

    - Uses streaming to handle large files.
    - Skips download if the file already exists.
    - Cleans up partially written files on failure.
    """
    url = rec.get("file_url")
    if not url:
        return False, 0

    filename = f"{rec.get('id')}_q{rec.get('quality')}.mp3"
    filepath = species_dir / filename

    # If the file already exists, assume it is complete and skip downloading.
    if filepath.exists():
        try:
            return True, filepath.stat().st_size
        except OSError:
            # If stat fails, try downloading again.
            pass

    client = session or requests
    try:
        with client.get(url, stream=True, timeout=60) as r:
            if r.status_code != 200:
                return False, 0

            downloaded = 0
            with filepath.open("wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if not chunk:
                        continue
                    f.write(chunk)
                    downloaded += len(chunk)
        return True, downloaded
    except requests.RequestException:
        # Remove incomplete file on error.
        if filepath.exists():
            try:
                filepath.unlink()
            except OSError:
                pass
        return False, 0


def collect_download_jobs(species_list: List[str]) -> List[Tuple[Dict, Path]]:
    """
    Collect all download jobs based on metadata files.

    For each species we:
    - read its metadata JSON
    - create a target directory under `audio_data/`
    - add a job for each recording that does not yet exist on disk
    """
    jobs: List[Tuple[Dict, Path]] = []
    metadata_dir = Path("metadata")

    for meta_file in metadata_dir.glob("*.json"):
        species_name = meta_file.stem.replace("_metadata", "").replace("_", " ")
        if species_name not in species_list:
            continue

        dest_dir = Path("audio_data") / meta_file.stem.replace("_metadata", "")
        dest_dir.mkdir(parents=True, exist_ok=True)

        try:
            with meta_file.open("r", encoding="utf-8") as f:
                records = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:  # noqa: BLE001
            log(f"Failed to read '{meta_file}': {exc}")
            continue

        for rec in records:
            filename = f"{rec.get('id')}_q{rec.get('quality')}.mp3"
            target = dest_dir / filename
            if not target.exists():
                jobs.append((rec, dest_dir))

    return jobs


def download_jobs_with_progress(jobs: List[Tuple[Dict, Path]], max_workers: int) -> None:
    """
    Download all jobs in parallel with a progress + throughput display.

    The progress line shows:
    - number of completed downloads
    - percentage of all jobs
    - total MB downloaded
    - average MB/s since start
    """
    total = len(jobs)
    if total == 0:
        log("No new files to download.")
        return

    log(f"Downloading {total} filtered files...")
    done = 0
    total_bytes = 0
    start_time = time.time()

    with requests.Session() as session:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(download_single_file, rec, dest, session)
                for rec, dest in jobs
            ]

            for idx, future in enumerate(as_completed(futures), start=1):
                success, size = future.result()
                if success:
                    done += 1
                    total_bytes += size

                # Update progress every 5 files or at the very end.
                if idx % 5 == 0 or idx == total:
                    elapsed = max(time.time() - start_time, 1e-6)
                    mb = total_bytes / (1024 * 1024)
                    speed = mb / elapsed
                    pct = idx / total * 100.0

                    print(
                        f"Status: {done}/{total} ({pct:.1f}%) | "
                        f"Total: {mb:.1f} MB | "
                        f"Speed: {speed:.2f} MB/s   ",
                        end="\r",
                        flush=True,
                    )

    print()  # Newline after the progress bar
    log(f"Done! {done} files downloaded.")


# ==============================================================================
# MAIN FLOW
# ==============================================================================


def collect_metadata_for_species(species: List[str]) -> None:
    """
    Collect metadata for all species in parallel.

    Uses either:
    - `search_unlimited`  (DOWNLOAD_MODE == "UNLIMITED")
    - `search_limited`    (otherwise)
    """
    if not species:
        log("No species found, exiting.")
        return

    log("--- Collecting metadata ---")
    search_fn = search_unlimited if DOWNLOAD_MODE.upper() == "UNLIMITED" else search_limited

    with ThreadPoolExecutor(max_workers=SEARCH_WORKERS) as executor:
        future_to_species = {
            executor.submit(
                search_fn,
                s,
                *( [LIMIT_PER_CATEGORY] if search_fn is search_limited else [] ),
            ): s
            for s in species
        }

        for future in as_completed(future_to_species):
            species_name = future_to_species[future]
            try:
                name, recordings = future.result()
                save_metadata(name, recordings)
            except Exception as exc:  # noqa: BLE001
                log(f"Error while collecting metadata for '{species_name}': {exc}")


def main() -> None:
    """Entry point for the Xeno‑canto downloader."""
    log("=" * 60)
    log("XENO-CANTO DOWNLOADER (Speed Display)")
    log(f"Filter: <= {MAX_DURATION_SECONDS}s | Mode: {DOWNLOAD_MODE}")
    log("=" * 60)

    species = load_species_from_csv(CSV_FILENAME, START_INDEX, END_INDEX)
    if not species:
        sys.exit(0)

    # Phase 1: metadata collection.
    collect_metadata_for_species(species)

    # Phase 2: audio downloads.
    log("--- Downloads ---")
    jobs = collect_download_jobs(species)
    download_jobs_with_progress(jobs, DOWNLOAD_WORKERS)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Interrupted by user.")
        os._exit(1)
