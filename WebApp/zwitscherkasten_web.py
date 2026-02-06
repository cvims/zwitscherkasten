#!/usr/bin/env python3
"""
🐦 Zwitscherkasten - Web-Dashboard mit Live-Erkennung
=====================================================
Design basiert auf dem offiziellen Zwitscherkasten Design-System.
Logo wird aus static/logo.svg geladen.
"""

import os
import sys
import time
import queue
import threading
import argparse
from datetime import datetime
from collections import deque
from dataclasses import dataclass, asdict
from typing import List, Optional
import numpy as np

try:
    from flask import Flask, render_template_string, jsonify, send_from_directory
    from flask_socketio import SocketIO
except ImportError:
    print("❌ Flask nicht installiert!")
    print("   pip install flask flask-socketio")
    sys.exit(1)

try:
    import sounddevice as sd
except ImportError:
    print("❌ sounddevice nicht installiert!")
    print("   pip install sounddevice")
    sys.exit(1)

from zwitscherkasten_inference import Zwitscherkasten, Config


# ============================================================================
# KONFIGURATION
# ============================================================================

SAMPLE_RATE = 48000
CHUNK_DURATION = 3.0
CHUNK_OVERLAP = 0.5
INTENT_THRESHOLD = 0.80
SPECIES_THRESHOLD = 0.70
COOLDOWN_SECONDS = 3.0
MAX_HISTORY = 500


# ============================================================================
# DATENSTRUKTUREN
# ============================================================================

@dataclass
class Detection:
    timestamp: str
    species: str
    species_scientific: str
    confidence: float
    intent_confidence: float
    processing_time_ms: float
    
    def to_dict(self):
        return asdict(self)


# ============================================================================
# AUDIO-PROCESSOR
# ============================================================================

class AudioProcessor:
    def __init__(self, socketio, device=None):
        self.socketio = socketio
        self.device = device
        self.running = False
        self.chunk_samples = int(SAMPLE_RATE * CHUNK_DURATION)
        self.hop_samples = int(self.chunk_samples * (1 - CHUNK_OVERLAP))
        self.audio_buffer = deque(maxlen=self.chunk_samples * 2)
        self.audio_queue = queue.Queue(maxsize=5)
        self.detections = []
        self.last_detection_time = {}
        self.audio_level = 0.0
        self.stats = {
            'start_time': None,
            'total_chunks': 0,
            'intent_triggers': 0,
            'species_today': set(),
            'detections_today': 0
        }
        
        print("\n🔧 Lade Modelle...")
        zk_config = Config()
        zk_config.intent_threshold = INTENT_THRESHOLD
        zk_config.top_k = 3
        self.zk = Zwitscherkasten(zk_config)
        print(f"✅ Audio-Processor bereit!")
        print(f"   Intent-Schwelle: {INTENT_THRESHOLD*100:.0f}%")
        print(f"   Species-Schwelle: {SPECIES_THRESHOLD*100:.0f}%")

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"⚠️ Audio: {status}")
        audio = indata[:, 0].copy()
        self.audio_level = float(np.abs(audio).mean() * 100)
        self.audio_buffer.extend(audio)
        if len(self.audio_buffer) >= self.chunk_samples:
            chunk = np.array(list(self.audio_buffer)[:self.chunk_samples])
            for _ in range(self.hop_samples):
                if self.audio_buffer:
                    self.audio_buffer.popleft()
            try:
                self.audio_queue.put_nowait(chunk)
            except queue.Full:
                pass

    def _process_loop(self):
        while self.running:
            try:
                chunk = self.audio_queue.get(timeout=1.0)
                self._process_chunk(chunk)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Fehler: {e}")

    def _process_chunk(self, audio):
        self.stats['total_chunks'] += 1
        start = time.perf_counter()
        
        is_bird, intent_conf = self.zk.intent_detector.predict(audio, SAMPLE_RATE)
        if intent_conf < INTENT_THRESHOLD:
            return
        
        self.stats['intent_triggers'] += 1
        predictions = self.zk.species_classifier.predict(audio, SAMPLE_RATE)
        process_time = (time.perf_counter() - start) * 1000
        
        if not predictions:
            return
        
        top_pred = predictions[0]
        if top_pred['confidence'] < SPECIES_THRESHOLD:
            return
        
        species = top_pred['species']
        now = time.time()
        if species in self.last_detection_time:
            if now - self.last_detection_time[species] < COOLDOWN_SECONDS:
                return
        
        self.last_detection_time[species] = now
        detection = Detection(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            species=species,
            species_scientific=top_pred['species_scientific'],
            confidence=top_pred['confidence'],
            intent_confidence=intent_conf,
            processing_time_ms=process_time
        )
        
        self.detections.insert(0, detection)
        if len(self.detections) > MAX_HISTORY:
            self.detections.pop()
        
        self.stats['species_today'].add(species)
        self.stats['detections_today'] += 1
        print(f"🐦 {detection.timestamp} | {species} | {detection.confidence:.0%}")

    def start(self):
        if self.running:
            return
        self.running = True
        self.stats['start_time'] = datetime.now()
        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.process_thread.start()
        self.stream = sd.InputStream(
            device=self.device,
            channels=1,
            samplerate=SAMPLE_RATE,
            blocksize=int(SAMPLE_RATE * 0.1),
            callback=self._audio_callback
        )
        self.stream.start()
        print(f"\n🎤 Audio-Stream gestartet!")

    def stop(self):
        self.running = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        print("⏹️ Audio-Stream gestoppt")

    def get_stats(self):
        uptime = ""
        if self.stats['start_time']:
            delta = datetime.now() - self.stats['start_time']
            hours, remainder = divmod(int(delta.total_seconds()), 3600)
            minutes, seconds = divmod(remainder, 60)
            uptime = f"{hours}h {minutes}min {seconds}s"
        return {
            'running': self.running,
            'uptime': uptime,
            'total_chunks': self.stats['total_chunks'],
            'intent_triggers': self.stats['intent_triggers'],
            'detections_today': self.stats['detections_today'],
            'species_today': len(self.stats['species_today']),
            'audio_level': self.audio_level
        }


# ============================================================================
# FLASK APP
# ============================================================================

app = Flask(__name__)
app.config['SECRET_KEY'] = 'zwitscherkasten-secret'
socketio = SocketIO(app, cors_allowed_origins="*")
processor = None

# Statische Dateien (Logo)
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(os.path.join(os.path.dirname(__file__), 'static'), filename)


HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Zwitscherkasten</title>
<style>
:root {
    --brand: #0D5961;
    --accent: #FF7359;
    --bg: #F5F7F5;
    --text: #1A2626;
    --gray: #667373;
    --white: #FFFFFF;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Rounded', sans-serif;
    background: var(--bg);
    min-height: 100vh;
    color: var(--text);
    padding: 20px;
}
.container { max-width: 900px; margin: 0 auto; }

/* Header */
header { text-align: center; padding: 25px 0; }
.logo-row { display: flex; align-items: center; justify-content: center; gap: 15px; margin-bottom: 8px; }
.logo { width: 55px; height: 55px; }
header h1 { font-size: 2.2em; font-weight: 800; color: var(--text); }
header p { color: var(--gray); font-size: 0.95em; }

/* Status Bar */
.status-bar {
    background: var(--white);
    border-radius: 22px;
    padding: 18px;
    margin-bottom: 18px;
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}
.status-item { text-align: center; flex: 1; min-width: 70px; }
.status-item .label {
    font-size: 0.7em;
    color: var(--gray);
    text-transform: uppercase;
    font-weight: 600;
    margin-bottom: 4px;
}
.status-item .value { font-size: 1.1em; font-weight: 700; }
.status-indicator {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    margin-right: 6px;
    background: var(--accent);
    animation: pulse 2s infinite;
}
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.5; } }

/* Audio Meter */
.audio-meter {
    background: var(--white);
    border-radius: 22px;
    padding: 15px 20px;
    margin-bottom: 18px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}
.audio-meter-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
}
.audio-meter-title { font-weight: 700; font-size: 0.85em; color: var(--text); }
.audio-meter-value { font-size: 0.8em; color: var(--brand); font-weight: 600; }
.audio-bars { display: flex; align-items: flex-end; gap: 4px; height: 35px; }
.audio-bar {
    flex: 1;
    background: var(--bg);
    border-radius: 3px;
    transition: height 0.15s, background 0.15s;
    min-height: 4px;
}
.audio-bar.active { background: var(--brand); }
.audio-bar.hot { background: var(--accent); }

/* Stats Grid */
.stats-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 15px;
    margin-bottom: 18px;
}
.stat-card {
    background: var(--white);
    border-radius: 22px;
    padding: 18px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}
.stat-card .number { font-size: 2em; font-weight: 800; color: var(--brand); }
.stat-card .label { font-size: 0.75em; color: var(--gray); margin-top: 4px; font-weight: 500; }

/* Detections */
.detections-container {
    background: var(--white);
    border-radius: 22px;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}
.detections-header {
    background: var(--brand);
    color: var(--white);
    padding: 15px 20px;
    font-weight: 700;
    display: flex;
    justify-content: space-between;
}
.detections-list { max-height: 400px; overflow-y: auto; }
.detection {
    padding: 14px 20px;
    border-bottom: 1px solid var(--bg);
    display: flex;
    align-items: center;
    gap: 14px;
}
.detection:hover { background: var(--bg); }
.detection-icon {
    width: 42px;
    height: 42px;
    background: var(--bg);
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.4em;
}
.detection-info { flex: 1; }
.detection-species { font-weight: 700; font-size: 0.95em; }
.detection-scientific { font-size: 0.75em; color: var(--gray); font-style: italic; }
.detection-time { color: var(--gray); font-size: 0.8em; font-weight: 500; }
.detection-confidence .percent { font-weight: 700; color: var(--brand); }
.confidence-bar { width: 70px; height: 5px; background: var(--bg); border-radius: 3px; margin-top: 4px; }
.confidence-fill { height: 100%; background: var(--accent); border-radius: 3px; }
.empty-state { text-align: center; padding: 50px 20px; color: var(--gray); }
.empty-state .icon { font-size: 3em; opacity: 0.4; margin-bottom: 10px; }

footer { text-align: center; padding: 25px; color: var(--gray); font-size: 0.8em; }
</style>
</head>
<body>
<div class="container">
<header>
<div class="logo-row">
<img src="/static/logo.svg" alt="Logo" class="logo">
<h1>Zwitscherkasten</h1>
</div>
<p>Vogelstimmenerkennung in Echtzeit</p>
</header>

<div class="status-bar">
<div class="status-item"><div class="label">Status</div><div class="value"><span class="status-indicator"></span>Aktiv</div></div>
<div class="status-item"><div class="label">Laufzeit</div><div class="value" id="uptime">--</div></div>
<div class="status-item"><div class="label">Letzter Check</div><div class="value" id="lastCheck">--</div></div>
<div class="status-item"><div class="label">Chunks</div><div class="value" id="totalChunks">0</div></div>
</div>

<div class="audio-meter">
<div class="audio-meter-header">
<span class="audio-meter-title">🎤 Mikrofon-Pegel</span>
<span class="audio-meter-value" id="audioValue">0%</span>
</div>
<div class="audio-bars" id="audioBars"></div>
</div>

<div class="stats-grid">
<div class="stat-card"><div class="number" id="detectionsToday">0</div><div class="label">Erkennungen</div></div>
<div class="stat-card"><div class="number" id="speciesToday">0</div><div class="label">Arten</div></div>
<div class="stat-card"><div class="number" id="intentTriggers">0</div><div class="label">Trigger</div></div>
</div>

<div class="detections-container">
<div class="detections-header"><span>Erkennungen</span><span id="detectionCount">0</span></div>
<div class="detections-list" id="detectionsList">
<div class="empty-state"><div class="icon">🎤</div><p>Warte auf Vogelstimmen...</p></div>
</div>
</div>

<footer>Zwitscherkasten &middot; TH Ingolstadt</footer>
</div>

<script>
var detections = [];
var barCount = 30;
var barsHtml = "";
for (var i = 0; i < barCount; i++) { barsHtml += "<div class='audio-bar'></div>"; }
document.getElementById("audioBars").innerHTML = barsHtml;

function loadData() {
    var x1 = new XMLHttpRequest();
    x1.open("GET", "/api/stats", true);
    x1.onreadystatechange = function() {
        if (x1.readyState == 4 && x1.status == 200) {
            var s = JSON.parse(x1.responseText);
            document.getElementById("uptime").textContent = s.uptime || "--";
            document.getElementById("detectionsToday").textContent = s.detections_today || 0;
            document.getElementById("speciesToday").textContent = s.species_today || 0;
            document.getElementById("intentTriggers").textContent = s.intent_triggers || 0;
            document.getElementById("totalChunks").textContent = s.total_chunks || 0;
            document.getElementById("lastCheck").textContent = new Date().toLocaleTimeString();
            updateAudioMeter(s.audio_level || 0);
        }
    };
    x1.send();

    var x2 = new XMLHttpRequest();
    x2.open("GET", "/api/detections", true);
    x2.onreadystatechange = function() {
        if (x2.readyState == 4 && x2.status == 200) {
            detections = JSON.parse(x2.responseText);
            renderDetections();
        }
    };
    x2.send();
}

function updateAudioMeter(level) {
    var bars = document.querySelectorAll(".audio-bar");
    var activeBars = Math.floor((level / 100) * barCount);
    document.getElementById("audioValue").textContent = Math.round(level) + "%";
    for (var i = 0; i < bars.length; i++) {
        var height = 4;
        if (i < activeBars) {
            height = 8 + Math.random() * 27;
            bars[i].classList.add("active");
            if (level > 50) bars[i].classList.add("hot");
            else bars[i].classList.remove("hot");
        } else {
            bars[i].classList.remove("active", "hot");
        }
        bars[i].style.height = height + "px";
    }
}

function renderDetections() {
    var list = document.getElementById("detectionsList");
    document.getElementById("detectionCount").textContent = detections.length;
    if (detections.length == 0) {
        list.innerHTML = "<div class='empty-state'><div class='icon'>🎤</div><p>Warte auf Vogelstimmen...</p></div>";
        return;
    }
    var html = "";
    for (var i = 0; i < detections.length; i++) {
        var d = detections[i];
        var conf = Math.round(d.confidence * 100);
        var time = d.timestamp.split(" ")[1];
        html += "<div class='detection'>";
        html += "<div class='detection-icon'>🐦</div>";
        html += "<div class='detection-info'>";
        html += "<div class='detection-species'>" + d.species + "</div>";
        html += "<div class='detection-scientific'>" + d.species_scientific + "</div>";
        html += "</div>";
        html += "<div class='detection-time'>" + time + "</div>";
        html += "<div class='detection-confidence'>";
        html += "<div class='percent'>" + conf + "%</div>";
        html += "<div class='confidence-bar'><div class='confidence-fill' style='width:" + conf + "%'></div></div>";
        html += "</div></div>";
    }
    list.innerHTML = html;
}

loadData();
setInterval(loadData, 2000);
</script>
</body>
</html>'''


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/detections')
def get_detections():
    if processor:
        return jsonify([d.to_dict() for d in processor.detections])
    return jsonify([])

@app.route('/api/stats')
def get_stats():
    if processor:
        return jsonify(processor.get_stats())
    return jsonify({})


# ============================================================================
# MAIN
# ============================================================================

def list_devices():
    print("\n🎤 Verfügbare Audio-Geräte:")
    print("-" * 50)
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            star = " ⭐" if i == sd.default.device[0] else ""
            print(f"   [{i}] {dev['name']}{star}")
    print("-" * 50)

def main():
    global processor
    parser = argparse.ArgumentParser(description="🐦 Zwitscherkasten Web-Dashboard")
    parser.add_argument('--port', '-p', type=int, default=5000)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--device', '-d', type=int, default=None)
    parser.add_argument('--list-devices', '-l', action='store_true')
    args = parser.parse_args()
    
    if args.list_devices:
        list_devices()
        return
    
    print("\n" + "="*60)
    print("🐦 ZWITSCHERKASTEN - Web-Dashboard")
    print("="*60)
    
    processor = AudioProcessor(socketio, device=args.device)
    processor.start()
    
    print(f"\n🌐 http://localhost:{args.port}")
    print(f"   Drücke Ctrl+C zum Beenden\n")
    
    try:
        socketio.run(app, host=args.host, port=args.port, debug=False, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        pass
    finally:
        processor.stop()
        print("\n👋 Auf Wiedersehen!\n")

if __name__ == "__main__":
    main()