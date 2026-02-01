"""
Zwitscherkasten - Bird Sound Recognition Web Application
=========================================================
Continuous monitoring mode:
- Laptop microphone records continuously
- Automatic bird detection and classification
- Web UI shows live results and history
"""

import os
import io
import json
import time
import threading
from datetime import datetime
from collections import deque
import numpy as np
import librosa
from flask import Flask, render_template, jsonify
from flask_cors import CORS
import tensorflow as tf
import onnxruntime as ort

# Fuer Mikrofon-Aufnahme
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
    print("+ sounddevice available for microphone recording")
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    print("- sounddevice not available - pip install sounddevice")

app = Flask(__name__)
CORS(app)

# === Configuration ===
SAMPLE_RATE = 16000
CHUNK_DURATION = 3  # Sekunden pro Analyse-Chunk
ANALYSIS_INTERVAL = 2  # Sekunden zwischen Analysen
HISTORY_MAX_SIZE = 100  # Maximale Historie-Eintraege

# Model paths
INTENT_MODEL_PATH = os.environ.get("INTENT_MODEL", "models/bird_intent_model.tflite")
CLASSIFICATION_MODEL_PATH = os.environ.get("CLASSIFICATION_MODEL", "models/model_audio.onnx")

# === Audio Processing Parameters ===
# Intent Model (aus preprocess_data.py)
INTENT_N_MELS = 64
INTENT_HOP_LENGTH = 512
INTENT_FMAX = 8000
INTENT_DURATION = 2.0  # Sekunden

# Classification Model (256 Klassen)
CLASS_N_MELS = 128
CLASS_HOP_LENGTH = 512
CLASS_N_FFT = 2048

# === Global State ===
BIRD_LABELS = None
intent_interpreter = None
classification_session = None

# Shared state fuer Thread-Kommunikation
current_result = {
    "timestamp": None,
    "is_bird": False,
    "intent_confidence": 0,
    "classification": None,
    "audio_level": 0
}
history = deque(maxlen=HISTORY_MAX_SIZE)
is_monitoring = False
monitor_thread = None
state_lock = threading.Lock()


def load_labels():
    """Lade Vogel-Labels aus JSON"""
    global BIRD_LABELS
    labels_path = "models/labels.json"
    if os.path.exists(labels_path):
        with open(labels_path, 'r', encoding='utf-8') as f:
            BIRD_LABELS = json.load(f)
        print(f"+ Loaded {len(BIRD_LABELS)} bird labels")
    else:
        BIRD_LABELS = [f"Bird_Species_{i}" for i in range(256)]
        print("- Using generic labels (no labels.json found)")


def load_models():
    """Lade beide Modelle"""
    global intent_interpreter, classification_session
    
    print(f"Loading Intent Model from {INTENT_MODEL_PATH}...")
    intent_interpreter = tf.lite.Interpreter(model_path=INTENT_MODEL_PATH)
    intent_interpreter.allocate_tensors()
    
    input_details = intent_interpreter.get_input_details()
    output_details = intent_interpreter.get_output_details()
    print(f"  + Intent Model loaded")
    print(f"    Input: {input_details[0]['shape']}")
    print(f"    Output: {output_details[0]['shape']}")
    
    print(f"Loading Classification Model from {CLASSIFICATION_MODEL_PATH}...")
    classification_session = ort.InferenceSession(CLASSIFICATION_MODEL_PATH)
    
    for inp in classification_session.get_inputs():
        print(f"  + Classification Model loaded")
        print(f"    Input: {inp.name}, Shape: {inp.shape}")
    for out in classification_session.get_outputs():
        print(f"    Output: {out.name}, Shape: {out.shape}")


def create_intent_spectrogram(audio):
    """
    Erstelle Mel-Spektrogramm fuer Intent Model [1, 64, 63, 1]
    WICHTIG: Exakt gleiche Parameter wie beim Training!
    """
    # Audio auf exakt 2 Sekunden bringen (wie beim Training)
    target_samples = int(SAMPLE_RATE * INTENT_DURATION)  # 32000 samples
    
    if len(audio) < target_samples:
        # Padding wie beim Training
        audio = np.pad(audio, (0, target_samples - len(audio)))
    else:
        # Nur die ersten 2 Sekunden
        audio = audio[:target_samples]
    
    # Mel-Spektrogramm mit exakt gleichen Parametern wie Training
    mel_spec = librosa.feature.melspectrogram(
        y=audio, 
        sr=SAMPLE_RATE, 
        n_mels=INTENT_N_MELS,
        hop_length=INTENT_HOP_LENGTH,
        fmax=INTENT_FMAX
    )
    
    # Log-Skalierung (dB)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # WICHTIG: Gleiche Normalisierung wie beim Training!
    mel_spec_norm = (mel_spec_db + 80) / 80
    mel_spec_norm = np.clip(mel_spec_norm, 0, 1)
    
    # Shape sollte (64, 63) sein - anpassen falls noetig
    target_frames = 63
    if mel_spec_norm.shape[1] < target_frames:
        pad_width = target_frames - mel_spec_norm.shape[1]
        mel_spec_norm = np.pad(mel_spec_norm, ((0, 0), (0, pad_width)), mode='constant')
    elif mel_spec_norm.shape[1] > target_frames:
        mel_spec_norm = mel_spec_norm[:, :target_frames]
    
    # Reshape: (64, 63) -> (1, 64, 63, 1)
    return mel_spec_norm.reshape(1, INTENT_N_MELS, target_frames, 1).astype(np.float32)


def create_classification_spectrogram(audio):
    """
    Erstelle Mel-Spektrogramm fuer Classification Model
    Input: [batch, 1, 128, time]
    """
    mel_spec = librosa.feature.melspectrogram(
        y=audio, 
        sr=SAMPLE_RATE, 
        n_mels=CLASS_N_MELS,
        n_fft=CLASS_N_FFT, 
        hop_length=CLASS_HOP_LENGTH
    )
    
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalisierung
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    
    # Reshape: (128, time) -> (1, 1, 128, time)
    return mel_spec_norm.reshape(1, 1, CLASS_N_MELS, -1).astype(np.float32)


def run_intent_detection(audio):
    """Intent Detection: Vogel ja/nein"""
    spectrogram = create_intent_spectrogram(audio)
    
    input_details = intent_interpreter.get_input_details()
    output_details = intent_interpreter.get_output_details()
    
    intent_interpreter.set_tensor(input_details[0]['index'], spectrogram)
    intent_interpreter.invoke()
    
    output = intent_interpreter.get_tensor(output_details[0]['index'])
    confidence = float(output[0][0])
    
    # Sigmoid falls nicht bereits angewandt (Werte ausserhalb 0-1)
    if confidence < 0 or confidence > 1:
        confidence = 1 / (1 + np.exp(-confidence))
    
    return confidence > 0.5, confidence


def run_classification(audio):
    """Bird Species Classification (256 Klassen)"""
    spectrogram = create_classification_spectrogram(audio)
    
    input_name = classification_session.get_inputs()[0].name
    outputs = classification_session.run(None, {input_name: spectrogram})
    
    logits = outputs[0][0]  # Shape: (256,)
    
    # Softmax
    exp_logits = np.exp(logits - np.max(logits))
    probabilities = exp_logits / exp_logits.sum()
    
    # Top-5 Ergebnisse
    top_indices = np.argsort(probabilities)[::-1]
    
    results = []
    for idx in top_indices[:5]:
        species = BIRD_LABELS[idx] if idx < len(BIRD_LABELS) else f"Unknown_{idx}"
        results.append({
            "species": species,
            "probability": float(probabilities[idx]),
            "index": int(idx)
        })
    
    return results


def analyze_audio(audio):
    """Analysiere Audio-Chunk"""
    global current_result, history
    
    # Normalisieren
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    
    # Audio-Level berechnen (RMS)
    audio_level = float(np.sqrt(np.mean(audio**2)))
    
    # Intent Detection
    is_bird, intent_confidence = run_intent_detection(audio)
    
    timestamp = datetime.now().isoformat()
    
    result = {
        "timestamp": timestamp,
        "is_bird": is_bird,
        "intent_confidence": round(intent_confidence, 4),
        "classification": None,
        "audio_level": round(audio_level, 4)
    }
    
    # Classification nur wenn Vogel erkannt
    if is_bird:
        classifications = run_classification(audio)
        result["classification"] = classifications
        
        # Zur Historie hinzufuegen
        history_entry = {
            "timestamp": timestamp,
            "species": classifications[0]["species"],
            "probability": classifications[0]["probability"],
            "top_3": classifications[:3]
        }
        with state_lock:
            history.appendleft(history_entry)
        
        print(f"[BIRD] [{timestamp[11:19]}] {classifications[0]['species']} ({classifications[0]['probability']*100:.1f}%)")
    else:
        print(f"[----] [{timestamp[11:19]}] No bird (conf: {intent_confidence:.2f}, level: {audio_level:.3f})")
    
    with state_lock:
        current_result.update(result)
    
    return result


def monitoring_loop():
    """Haupt-Loop fuer kontinuierliche Mikrofon-Ueberwachung"""
    global is_monitoring
    
    print(f"\n[MIC] Starting continuous monitoring...")
    print(f"   Sample Rate: {SAMPLE_RATE} Hz")
    print(f"   Chunk Duration: {CHUNK_DURATION}s")
    print(f"   Analysis Interval: {ANALYSIS_INTERVAL}s\n")
    
    chunk_samples = int(SAMPLE_RATE * CHUNK_DURATION)
    
    while is_monitoring:
        try:
            # Audio aufnehmen
            audio = sd.rec(chunk_samples, samplerate=SAMPLE_RATE, channels=1, dtype='float32')
            sd.wait()
            audio = audio.flatten()
            
            # Analysieren
            analyze_audio(audio)
            
            # Pause vor naechster Analyse
            time.sleep(ANALYSIS_INTERVAL)
            
        except Exception as e:
            print(f"[ERROR] in monitoring loop: {e}")
            time.sleep(1)
    
    print("[STOP] Monitoring stopped")


def start_monitoring():
    """Starte Monitoring in separatem Thread"""
    global is_monitoring, monitor_thread
    
    if not SOUNDDEVICE_AVAILABLE:
        return False, "sounddevice nicht installiert"
    
    if is_monitoring:
        return False, "Monitoring laeuft bereits"
    
    is_monitoring = True
    monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
    monitor_thread.start()
    return True, "Monitoring gestartet"


def stop_monitoring():
    """Stoppe Monitoring"""
    global is_monitoring
    is_monitoring = False
    return True, "Monitoring gestoppt"


# === Flask Routes ===

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/status')
def get_status():
    """Aktueller Status und letzte Erkennung"""
    with state_lock:
        return jsonify({
            "monitoring": is_monitoring,
            "current": current_result.copy(),
            "sounddevice_available": SOUNDDEVICE_AVAILABLE
        })


@app.route('/api/history')
def get_history():
    """Historie aller Erkennungen"""
    with state_lock:
        return jsonify({
            "count": len(history),
            "entries": list(history)
        })


@app.route('/api/start', methods=['POST'])
def api_start():
    """Starte Monitoring"""
    success, message = start_monitoring()
    return jsonify({"success": success, "message": message})


@app.route('/api/stop', methods=['POST'])
def api_stop():
    """Stoppe Monitoring"""
    success, message = stop_monitoring()
    return jsonify({"success": success, "message": message})


@app.route('/api/clear', methods=['POST'])
def api_clear():
    """Loesche Historie"""
    with state_lock:
        history.clear()
    return jsonify({"success": True, "message": "Historie geloescht"})


@app.route('/api/health')
def health():
    return jsonify({
        "status": "healthy",
        "models_loaded": intent_interpreter is not None,
        "sounddevice_available": SOUNDDEVICE_AVAILABLE,
        "monitoring": is_monitoring,
        "num_classes": len(BIRD_LABELS) if BIRD_LABELS else 0
    })


# === Main ===

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--https', action='store_true', help='Enable HTTPS')
    parser.add_argument('--autostart', action='store_true', help='Auto-start monitoring')
    args = parser.parse_args()
    
    load_labels()
    load_models()
    
    # SSL-Kontext
    ssl_context = None
    if args.https:
        if os.path.exists('cert.pem') and os.path.exists('key.pem'):
            ssl_context = ('cert.pem', 'key.pem')
            print("+ HTTPS aktiviert")
        else:
            print("- Zertifikate nicht gefunden!")
            exit(1)
    
    # Auto-Start Monitoring
    if args.autostart:
        start_monitoring()
    
    protocol = "https" if ssl_context else "http"
    
    print("\n" + "="*60)
    print("Zwitscherkasten Server")
    print("="*60)
    print(f"Local:  {protocol}://localhost:5000")
    print(f"LAN:    {protocol}://192.168.2.122:5000")
    print("")
    print("Commands:")
    print("  python app.py              Normal start")
    print("  python app.py --autostart  Auto-start monitoring")
    print("  python app.py --https      Enable HTTPS (for iPhone)")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False, ssl_context=ssl_context)
