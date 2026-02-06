# Zwitscherkasten — WebApp

Short description: This WebApp provides a simple inference interface for the audio models included in this repository. The entry point is the script `zwitscherkasten_web.py`.

**Overview**
- **Purpose:** Local web service for audio inference (bird vocalization recognition / intent detection).
- **Entry point:** `zwitscherkasten_web.py`

**Important files (in the `WebApp` folder)**
- `zwitscherkasten_web.py`: Starts the web server and exposes HTTP endpoint(s).
- `raspi_realtime.py`, `zwitscherkasten_inference.py`, `zwitscherkasten_live.py`: Helper scripts for local inference and live setups.
- `efficientnet_b0_audio.onnx`, `bird_intent_model.tflite`, `class_map.json`, `model_metadata.json`: Provided models and metadata.

## Usage

1. run `zwitscherkasten_web.py`
2. Open the app in a browser: `http://localhost:5000` — or the host/port printed by `zwitscherkasten_web.py`.
