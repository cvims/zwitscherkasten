# 🐦 Zwitscherkasten — Audio Intent (Audio_Intent)

This folder contains tools and a reference SampleApp for audio-based bird intent detection and species classification.

Summary of what changed: the web/demo application and model files are now located in the `SampleApp/` subfolder. Top-level scripts such as `preprocess_data.py` and `train_model.py` are the utilities used for data preparation and model training.

## Layout (current)

Audio_Intent/
- SampleApp/
    - `app.py`                # Reference Flask app + live monitoring UI
    - `bird_intent_model.tflite`  # TFLite intent model (provided)
    - `model_audio.onnx`      # ONNX classification model (provided)
    - `generate_cert.py`      # Helper to create `cert.pem` / `key.pem` for HTTPS
    - `model_audio.onnx.data` # auxiliary model weight/data if present
- `labels.json`             # Species label map (256 classes)
- `preprocess_data.py`      # Convert raw audio → mel-spectrograms used for training
- `train_model.py`          # Training script for intent/classifier models
- `requirements.txt`       # Python dependencies for the Audio_Intent tools

## Quick start (run the sample app)

1. From repository root, install dependencies:

```bash
cd Audio_Intent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the SampleApp (models are shipped in `SampleApp/`):

```bash
cd SampleApp
python app.py
```

3. Optional flags (see `SampleApp/app.py`):

```bash
python app.py --autostart   # start monitoring automatically
python app.py --https       # use HTTPS (requires cert.pem/key.pem)
```

Notes:
- `app.py` expects `models/` by default (environment variables `INTENT_MODEL` and `CLASSIFICATION_MODEL` can override paths). Place `bird_intent_model.tflite` and `model_audio.onnx` into a `SampleApp/models/` directory or set environment variables before launching.
- For local quick testing you can also run with the models present directly in `SampleApp/` and set `INTENT_MODEL` and `CLASSIFICATION_MODEL` accordingly:

```bash
export INTENT_MODEL=./SampleApp/bird_intent_model.tflite
export CLASSIFICATION_MODEL=./SampleApp/model_audio.onnx
python SampleApp/app.py
```

## Configuration (reference values used by `SampleApp/app.py`)

- `SAMPLE_RATE = 16000`
- `CHUNK_DURATION = 3` (seconds)
- `ANALYSIS_INTERVAL = 2` (seconds between analyses)
- `HISTORY_MAX_SIZE = 100`

## API Endpoints (provided by `SampleApp/app.py`)

- `/` — GET: Web UI (templates under `SampleApp/templates/` if present)
- `/api/status` — GET: current monitoring status and last result
- `/api/history` — GET: detection history
- `/api/start` — POST: start monitoring
- `/api/stop` — POST: stop monitoring
- `/api/clear` — POST: clear history
- `/api/health` — GET: health and model status

## Data preparation & training

- Use `preprocess_data.py` to convert raw audio files into mel-spectrogram inputs compatible with the intent and classification models.
- Use `train_model.py` to train or fine-tune models; consult the script headers for training parameters.

## Notes for deployment

- For Raspberry Pi / edge deployment prefer `tflite-runtime` for the intent model and a lightweight ONNX runtime for classification.
- If you need HTTPS for mobile access, run `SampleApp/generate_cert.py` (or provide your own certs) and start with `--https`.

## Next steps

- I can:
    - Move the provided models into a `SampleApp/models/` subfolder and set defaults in `app.py`.
    - Generate a minimal `SampleApp/.env` or startup script that sets `INTENT_MODEL` / `CLASSIFICATION_MODEL`.

---
If you want, I will apply one of the next steps above (create `SampleApp/models/` and update `app.py` defaults, or add a `.env` / run script). 
