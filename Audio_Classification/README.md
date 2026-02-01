# ZwitscherkastenVectrex 🐦🎵

**ZwitscherkastenVectrex** is a small toolkit for collecting bird audio from Xeno‑Canto, preparing datasets, visualizing audio, and training audio classification models (EfficientNet / MobileNet / PaSST) for bird species recognition.

---

## Features

- Collect and filter bird audio from Xeno‑Canto with configurable quality and duration limits. (`DataProcessing`)
- Save per-species metadata and download audio into organized directories. (`sampled_metadata/`, `audio_data/`)
- Prepare datasets and training pipelines for multiple architectures: EfficientNet B0/B3, MobileNet, and PaSST. (`Training/`)
- Utilities for plotting mel-spectrograms and exporting models to ONNX. (`DataProcessing/`, `Tools/`)

---

## Repository Structure

- `DataProcessing/` — scripts to search, filter, and download audio files; plot mel-spectrograms.
  - `download_audio_data.py` — batch downloader and metadata fetcher.
  - `download_random_audio.py` — quick sampler selecting random species from CSV.
  - `plot_mels.py` — helpers to visualize audio.
  - `Vogel_Vergleich_Liste_Birdset_vs_XenoCanto.csv` — species mapping CSV used by the downloaders.

- `Training/` — training code and dataset preparation per model family.
  - `EfficientNet_B0_256Classes/`, `EfficientNet_B3_256Classes/`, `MobileNet_256Classes/`, `PaSST_256Classes/`
  - Each contains `dataset.py`, `prepare_data_from_raw_audio.py`, and `train_*.py` scripts.

- `Tools/` — small utilities
  - `CUDA_checker.py` — verify CUDA availability.
  - `ONNX_exporter.py` — export trained models to ONNX.

- `requirements.txt` — Python dependencies.

---

## 🧰 Training

1. Prepare data (per-model helpers exist in each training subfolder):

```bash
python Training/EfficientNet_B0_256Classes/prepare_data_from_raw_audio.py
```

2. Train a model (example):

```bash
python Training/EfficientNet_B0_256Classes/train_efficientnet.py
```

Check the individual training folders for CLI arguments, dataset expectations, and hyperparameters.

