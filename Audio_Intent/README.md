# 🐦 Zwitscherkasten

**Real-time bird sound recognition for edge devices**

Zwitscherkasten is a lightweight, multimodal bird monitoring system that performs audio-based species recognition on low-power edge hardware such as Raspberry Pi. It uses a two-stage ML pipeline for efficient detection and classification of 256 European bird species.

![Zwitscherkasten Demo](monitoring.png)

## ✨ Features

- **Two-Stage Detection Pipeline**: Intent model (is it a bird?) → Classification model (which species?)
- **Real-time Monitoring**: Continuous audio analysis via microphone
- **Web Interface**: Live dashboard accessible from any device on your network
- **Edge-Optimized**: TFLite + ONNX models designed for Raspberry Pi performance
- **256 Species**: Covers most European bird species with scientific naming
- **HTTPS Support**: Secure access from mobile devices

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│   Microphone    │────▶│  Intent Model    │────▶│ Classification Model│
│   (3s chunks)   │     │  (TFLite, 12KB)  │     │   (ONNX, 6.6MB)     │
└─────────────────┘     │  Bird? Yes/No    │     │   256 species       │
                        └──────────────────┘     └─────────────────────┘
                                                           │
                                                           ▼
                                                  ┌─────────────────┐
                                                  │   Flask Web UI  │
                                                  │   Live Results  │
                                                  └─────────────────┘
```

The intent model acts as a lightweight gate, preventing unnecessary classification inference when no bird sounds are detected.

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Working microphone
- ~100MB RAM

### Installation

```bash
git clone https://github.com/cvims/zwitscherkasten.git
cd zwitscherkasten
pip install -r requirements.txt
```

### Run

```bash
# Standard start
python app.py

# Auto-start monitoring
python app.py --autostart

# Enable HTTPS (required for mobile access)
python app.py --https
```

Open `http://localhost:5000` in your browser.

## 📱 Mobile Access

For iPhone/Android access over your local network:

```bash
# Generate SSL certificates (first time only)
python generate_cert.py

# Start with HTTPS
python app.py --https
```

Then access via `https://<your-ip>:5000`

## 🧠 Models

| Model | Format | Size | Purpose |
|-------|--------|------|---------|
| `bird_intent_model.tflite` | TensorFlow Lite | 12 KB | Binary bird detection |
| `model_audio.onnx` | ONNX | 6.6 MB | Species classification (256 classes) |

### Audio Processing

- **Sample Rate**: 16 kHz
- **Chunk Duration**: 3 seconds
- **Features**: Mel spectrograms (64/128 bands)

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/api/status` | GET | Current detection status |
| `/api/history` | GET | Detection history |
| `/api/start` | POST | Start monitoring |
| `/api/stop` | POST | Stop monitoring |
| `/api/clear` | POST | Clear history |
| `/api/health` | GET | System health check |

## 🛠️ Configuration

Environment variables:

```bash
INTENT_MODEL=models/bird_intent_model.tflite
CLASSIFICATION_MODEL=models/model_audio.onnx
```

Parameters in `app.py`:

```python
SAMPLE_RATE = 16000        # Audio sample rate
CHUNK_DURATION = 3         # Seconds per analysis
ANALYSIS_INTERVAL = 2      # Seconds between analyses
HISTORY_MAX_SIZE = 100     # Max history entries
```

## 🐧 Raspberry Pi Deployment

```bash
# Install system dependencies
sudo apt-get install libportaudio2 libsndfile1

# Install Python packages
pip install -r requirements.txt

# Run on startup (optional)
# Add to /etc/rc.local:
# python /home/pi/zwitscherkasten/app.py --autostart &
```

## 📦 Project Structure

```
zwitscherkasten/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── generate_cert.py       # SSL certificate generator
├── templates/
│   └── index.html         # Web interface
└── models/
    ├── bird_intent_model.tflite   # Intent detection
    ├── model_audio.onnx           # Species classification
    ├── model_audio.onnx.data      # ONNX weights
    └── labels.json                # 256 species labels
```

## 🎓 Academic Context

This project was developed as part of the Applied AI curriculum at Technische Hochschule Ingolstadt (THI) - Project AKI 2025.

## 👥 Contributors

- Florian Schulenberg ([@SirVectrex](https://github.com/SirVectrex))
- Fabian Jirges ([@MasterCodeMan96](https://github.com/MasterCodeMan96))

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

*Made with 🎵 for the birds*
