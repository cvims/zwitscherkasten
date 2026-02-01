"""
Preprocess Bird Audio Data to Mel Spectrograms
===============================================

Converts raw MP3/WAV/FLAC recordings into mel spectrograms (uint8) suitable
for model training. Handles class mapping, oversampling, and train/val split.

"""

import librosa
import numpy as np
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


# --- CONFIGURATION ---
CURRENT_DIR = Path(__file__).parent.resolve()

# Your structure:
# /ZwitscherkastenTransformerTrack/
#   ├── DataProcessing/
#   │   └── audio_data/          ← DATA
#   └── Training/
#       └── ModelType/           ← THIS SCRIPT    
#

AUDIO_ROOT = CURRENT_DIR.parent.parent / "DataProcessing" / "audio_data"
PREPROCESSED_OUTPUT = CURRENT_DIR / "preprocessed_mels"

print(f"\n{'='*70}")
print(f"Audio folder: {AUDIO_ROOT}")
print(f"{'='*70}\n")

if not AUDIO_ROOT.exists():
    print(f"❌ ERROR: Audio folder not found at {AUDIO_ROOT}")
    print(f"\nPlease check:")
    print(f"  1. Is your audio data in: ZwitscherkastenTransformerTrack/DataProcessing/audio_data/")
    print(f"  2. Does it have subdirectories for each species?")
    print(f"  3. Are there MP3 files inside the species folders?")
    exit(1)

# Audio preprocessing
TARGET_SR = 32000
N_MELS = 128
N_FFT = 512
HOP_LENGTH = 512

# Dataset config
VAL_SIZE = 0.1
MIN_SAMPLES_PER_CLASS = 500
SEED = 42


def preprocess_audio_to_mel(audio_path, target_sr=32000, n_mels=128, n_fft=512):
    """Convert MP3/WAV → Mel spectrogram → uint8"""
    try:
        y, sr = librosa.load(str(audio_path), sr=target_sr)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=HOP_LENGTH)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = np.clip(mel_db, -80, 0)
        mel_uint8 = ((mel_db + 80) / 80 * 255).astype(np.uint8)
        return mel_uint8
    except Exception as e:
        print(f"  ⚠️  Error loading {audio_path.name}: {e}")
        return None


def main():
    print(f"🎵 PREPROCESSING RAW AUDIO DATA\n")
    
    # Scan bird species
    bird_species = sorted([d for d in AUDIO_ROOT.iterdir() if d.is_dir()])
    
    if not bird_species:
        print(f"❌ ERROR: No species folders found in {AUDIO_ROOT}")
        return
    
    print(f"Found {len(bird_species)} bird species\n")
    
    # Create class mapping
    class_to_idx = {species.name: idx for idx, species in enumerate(bird_species)}
    idx_to_class = {str(v): k for k, v in class_to_idx.items()}
    
    with open(CURRENT_DIR / "class_map.json", "w") as f:
        json.dump(class_to_idx, f, indent=4)
    with open(CURRENT_DIR / "idx_to_class.json", "w") as f:
        json.dump(idx_to_class, f, indent=4)
    
    print(f"Species mapping:")
    for idx, species in enumerate(sorted(bird_species)):
        print(f"  {idx}: {species.name}")
    print()
    
    # Process audio files
    data_list = []
    total_files = 0
    successful_files = 0
    
    for species_folder in sorted(bird_species):
        species_name = species_folder.name
        species_idx = class_to_idx[species_name]
        
        output_species_dir = PREPROCESSED_OUTPUT / species_name
        output_species_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all audio files
        audio_files = list(species_folder.glob("*.mp3")) + \
                     list(species_folder.glob("*.wav")) + \
                     list(species_folder.glob("*.flac"))
        
        if not audio_files:
            print(f"⚠️  {species_name}: No audio files found")
            continue
        
        print(f"📍 {species_name}: Processing {len(audio_files)} files...")
        
        for audio_file in tqdm(audio_files, desc=f"  {species_name}", leave=False):
            total_files += 1
            mel_uint8 = preprocess_audio_to_mel(audio_file, TARGET_SR, N_MELS, N_FFT)
            
            if mel_uint8 is None:
                continue
            
            output_path = output_species_dir / f"{audio_file.stem}.npy"
            np.save(output_path, mel_uint8)
            
            data_list.append({
                "filepath": str(output_path.resolve()),
                "label": species_idx,
                "bird_name": species_name
            })
            successful_files += 1
        
        print(f"  ✓ {len(audio_files)} → {len([d for d in data_list if d['bird_name'] == species_name])} preprocessed\n")
    
    print(f"\n{'='*70}")
    print(f"PREPROCESSING COMPLETE")
    print(f"  Total found: {total_files}")
    print(f"  Successful: {successful_files}")
    print(f"  Total samples: {len(data_list)}")
    print(f"{'='*70}\n")
    
    if len(data_list) == 0:
        print("❌ No audio files were successfully processed!")
        return
    
    # Create dataframe
    df = pd.DataFrame(data_list)
    print(f"Original distribution:")
    print(df['bird_name'].value_counts().sort_index())
    print()
    
    # Stratified oversample
    print(f"Applying stratified oversampling...\n")
    oversampled_dfs = []
    
    for class_label in sorted(df['label'].unique()):
        class_df = df[df['label'] == class_label].reset_index(drop=True)
        class_name = idx_to_class[str(int(class_label))]
        current_count = len(class_df)
        
        if current_count < MIN_SAMPLES_PER_CLASS:
            class_df = resample(class_df, n_samples=MIN_SAMPLES_PER_CLASS, replace=True, random_state=SEED)
            print(f"  {class_name}: {current_count} → {MIN_SAMPLES_PER_CLASS}")
        else:
            print(f"  {class_name}: {current_count} (no oversample needed)")
        
        oversampled_dfs.append(class_df)
    
    df = pd.concat(oversampled_dfs, ignore_index=True)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    
    print(f"\n✅ After oversampling: {len(df)} total samples\n")
    
    # Split train/val
    train_df, val_df = train_test_split(df, test_size=VAL_SIZE, stratify=df['label'], random_state=SEED, shuffle=True)
    
    train_df.to_csv(CURRENT_DIR / "train.csv", index=False)
    val_df.to_csv(CURRENT_DIR / "val.csv", index=False)
    
    print(f"Train: {len(train_df)} | Val: {len(val_df)}\n")
    
    # Compute class weights
    class_counts = train_df['label'].value_counts().sort_index()
    num_classes = len(class_counts)
    class_weights = num_classes / (class_counts.values * len(class_counts))
    class_weight_dict = {str(int(idx)): float(weight) for idx, weight in enumerate(class_weights)}
    
    with open(CURRENT_DIR / "class_weights.json", "w") as f:
        json.dump(class_weight_dict, f, indent=4)
    
    print("Class weights:")
    for idx in sorted([int(k) for k in class_weight_dict.keys()]):
        bird_name = idx_to_class[str(idx)]
        print(f"  {bird_name}: {class_weight_dict[str(idx)]:.4f}")
    
    # Save stats
    with open(CURRENT_DIR / "passt_stats.json", "w") as f:
        json.dump({"mean": -4.2677393, "std": 4.5689974}, f, indent=4)
    
    summary = {
        "total_samples": len(df),
        "num_classes": num_classes,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "val_split": VAL_SIZE,
        "seed": SEED,
        "min_samples_per_class": MIN_SAMPLES_PER_CLASS,
        "preprocessed_dir": str(PREPROCESSED_OUTPUT),
        "class_distribution": {idx_to_class[str(int(idx))]: int(count) for idx, count in enumerate(class_counts)},
    }
    
    with open(CURRENT_DIR / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=4)
    
    print(f"\n{'='*70}")
    print("✅ ALL DONE!")
    print(f"{'='*70}")
    print(f"\nGenerated files:")
    print(f"  ✓ train.csv")
    print(f"  ✓ val.csv")
    print(f"  ✓ class_map.json")
    print(f"  ✓ class_weights.json")
    print(f"  ✓ dataset_summary.json")
    print(f"\nPreprocessed data: {PREPROCESSED_OUTPUT}/")
    print(f"\nNext: python train.py (set PHASE=1)")


if __name__ == "__main__":
    main()