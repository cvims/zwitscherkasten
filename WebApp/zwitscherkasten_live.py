#!/usr/bin/env python3
"""
🐦 Zwitscherkasten - Audio + Snapshot-Vision Dashboard
======================================================
- Audio-Erkennung über Mikrofon (kontinuierlich, wie bisher)
- Kamera-Snapshot per Button-Klick → Two-Stage Vision-Pipeline
- Historisierte Erkennungen (Audio + Vision) im Dashboard
- Optimiert für Raspberry Pi 5 (kein dauerhafter Video-Stream)
"""

import os
import sys
import io
import time
import json
import queue
import base64
import threading
import argparse
from datetime import datetime
from collections import deque
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Any
import numpy as np

try:
    from flask import Flask, render_template_string, jsonify, send_from_directory, Response, request
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

try:
    import cv2
except ImportError:
    print("❌ OpenCV nicht installiert!")
    print("   pip install opencv-python-headless")
    sys.exit(1)

from zwitscherkasten_inference import Zwitscherkasten, Config
from vision_split_pipeline import TwoStageONNX, draw_result


# ============================================================================
# KONFIGURATION
# ============================================================================

# Audio
SAMPLE_RATE = 48000
CHUNK_DURATION = 3.0
CHUNK_OVERLAP = 0.5
INTENT_THRESHOLD = 0.80
SPECIES_THRESHOLD = 0.70
COOLDOWN_SECONDS = 3.0

# Vision (Snapshot)
VISION_WIDTH = 1280
VISION_HEIGHT = 720

MAX_HISTORY = 500

# Picamera2 – importieren wenn verfügbar, sonst OpenCV-Fallback
PICAMERA_AVAILABLE = False
try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    print("⚠️  picamera2 nicht verfügbar – verwende OpenCV-Fallback")


# ============================================================================
# DEUTSCHE ARTNAMEN
# ============================================================================

GERMAN_NAMES = {
    "Turdus_merula": "Amsel",
    "Parus_major": "Kohlmeise",
    "Cyanistes_caeruleus": "Blaumeise",
    "Erithacus_rubecula": "Rotkehlchen",
    "Fringilla_coelebs": "Buchfink",
    "Passer_domesticus": "Haussperling",
    "Passer_montanus": "Feldsperling",
    "Sturnus_vulgaris": "Star",
    "Columba_palumbus": "Ringeltaube",
    "Streptopelia_decaocto": "Türkentaube",
    "Corvus_corone": "Rabenkrähe",
    "Pica_pica": "Elster",
    "Garrulus_glandarius": "Eichelhäher",
    "Dendrocopos_major": "Buntspecht",
    "Sitta_europaea": "Kleiber",
    "Aegithalos_caudatus": "Schwanzmeise",
    "Carduelis_carduelis": "Stieglitz",
    "Chloris_chloris": "Grünfink",
    "Phoenicurus_ochruros": "Hausrotschwanz",
    "Sylvia_atricapilla": "Mönchsgrasmücke",
    "Phylloscopus_collybita": "Zilpzalp",
    "Troglodytes_troglodytes": "Zaunkönig",
    "Motacilla_alba": "Bachstelze",
    "Anas_platyrhynchos": "Stockente",
    "Ardea_cinerea": "Graureiher",
    "Buteo_buteo": "Mäusebussard",
    "Falco_tinnunculus": "Turmfalke",
    "Corvus_corax": "Kolkrabe",
    "Corvus_frugilegus": "Saatkrähe",
    "Periparus_ater": "Tannenmeise",
    "Lophophanes_cristatus": "Haubenmeise",
    "Poecile_palustris": "Sumpfmeise",
    "Poecile_montanus": "Weidenmeise",
    "Regulus_regulus": "Wintergoldhähnchen",
    "Regulus_ignicapilla": "Sommergoldhähnchen",
    "Alauda_arvensis": "Feldlerche",
    "Emberiza_citrinella": "Goldammer",
    "Alcedo_atthis": "Eisvogel",
    "Ciconia_ciconia": "Weißstorch",
    "Apus_apus": "Mauersegler",
    "Hirundo_rustica": "Rauchschwalbe",
    "Delichon_urbicum": "Mehlschwalbe",
    "Pyrrhula_pyrrhula": "Gimpel",
    "Coccothraustes_coccothraustes": "Kernbeißer",
    "Linaria_cannabina": "Bluthänfling",
    "Spinus_spinus": "Erlenzeisig",
    "Luscinia_megarhynchos": "Nachtigall",
    "Serinus_serinus": "Girlitz",
    "Dryocopus_martius": "Schwarzspecht",
    "Picus_viridis": "Grünspecht",
    "Strix_aluco": "Waldkauz",
    "Tyto_alba": "Schleiereule",
    "Athene_noctua": "Steinkauz",
    "Cuculus_canorus": "Kuckuck",
    "Columba_livia_f._domestica": "Straßentaube",
    "Upupa_epops": "Wiedehopf",
    "Oriolus_oriolus": "Pirol",
    "Prunella_modularis": "Heckenbraunelle",
    "Turdus_philomelos": "Singdrossel",
    "Turdus_pilaris": "Wacholderdrossel",
    "Turdus_viscivorus": "Misteldrossel",
    "Turdus_iliacus": "Rotdrossel",
    "Corvus_cornix": "Nebelkrähe",
    "Coloeus_monedula": "Dohle",
    "Certhia_brachydactyla": "Gartenbaumläufer",
    "Certhia_familiaris": "Waldbaumläufer",
    "Acanthis_flammea": "Birkenzeisig",
    "Acanthis_cabaret": "Alpenbirkenzeisig",
    "Loxia_curvirostra": "Fichtenkreuzschnabel",
    "Aegolius_funereus": "Raufußkauz",
    "Bubo_bubo": "Uhu",
    "Asio_otus": "Waldohreule",
    "Accipiter_gentilis": "Habicht",
    "Grus_grus": "Kranich",
    "Vanellus_vanellus": "Kiebitz",
    "Numenius_arquata": "Großer Brachvogel",
    "Phoenicurus_phoenicurus": "Gartenrotschwanz",
    "Muscicapa_striata": "Grauschnäpper",
    "Ficedula_hypoleuca": "Trauerschnäpper",
    "Lanius_collurio": "Neuntöter",
    "Merops_apiaster": "Bienenfresser",
}


def get_german_name(scientific: str) -> str:
    """Gibt den deutschen Artnamen zurück (oder formatierten wissenschaftlichen)."""
    return GERMAN_NAMES.get(scientific, scientific.replace("_", " "))


# ============================================================================
# DATENSTRUKTUREN
# ============================================================================

@dataclass
class Detection:
    timestamp: str
    species: str
    species_scientific: str
    confidence: float
    source: str               # "audio" oder "vision"
    thumbnail_b64: str = ""   # Base64-JPEG (nur bei Vision)

    def to_dict(self):
        return asdict(self)


# ============================================================================
# AUDIO-PROCESSOR (unverändert aus zwitscherkasten_live.py)
# ============================================================================

class AudioProcessor:
    def __init__(self, on_detection, device=None):
        self.on_detection = on_detection
        self.device = device
        self.running = False
        self.chunk_samples = int(SAMPLE_RATE * CHUNK_DURATION)
        self.hop_samples = int(self.chunk_samples * (1 - CHUNK_OVERLAP))
        self.audio_buffer = deque(maxlen=self.chunk_samples * 2)
        self.audio_queue = queue.Queue(maxsize=5)
        self.last_detection_time = {}
        self.audio_level = 0.0
        self.stats = {'total_chunks': 0, 'intent_triggers': 0}

        print("\n🔧 Lade Audio-Modelle...")
        zk_config = Config()
        zk_config.intent_threshold = INTENT_THRESHOLD
        zk_config.top_k = 3
        self.zk = Zwitscherkasten(zk_config)
        print(f"✅ Audio-Processor bereit!")

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
                print(f"❌ Audio-Fehler: {e}")

    def _process_chunk(self, audio):
        self.stats['total_chunks'] += 1
        is_bird, intent_conf = self.zk.intent_detector.predict(audio, SAMPLE_RATE)
        if intent_conf < INTENT_THRESHOLD:
            return
        self.stats['intent_triggers'] += 1
        predictions = self.zk.species_classifier.predict(audio, SAMPLE_RATE)
        if not predictions:
            return
        top_pred = predictions[0]
        if top_pred['confidence'] < SPECIES_THRESHOLD:
            return
        species_sci = top_pred['species_scientific']
        species_name = get_german_name(species_sci)
        now = time.time()
        if species_name in self.last_detection_time:
            if now - self.last_detection_time[species_name] < COOLDOWN_SECONDS:
                return
        self.last_detection_time[species_name] = now
        detection = Detection(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            species=species_name,
            species_scientific=species_sci.replace('_', ' '),
            confidence=top_pred['confidence'],
            source="audio"
        )
        print(f"🎤 {detection.timestamp} | {species_name} | {detection.confidence:.0%}")
        self.on_detection(detection)

    def start(self):
        if self.running:
            return
        self.running = True
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
        print(f"🎤 Audio-Stream gestartet!")

    def stop(self):
        self.running = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()


# ============================================================================
# VISION-PROCESSOR (SNAPSHOT-BASIERT – NEU!)
# ============================================================================

class VisionProcessor:
    """
    Snapshot-basierte Kamera-Erkennung.
    Statt dauerhaftem Video-Stream wird nur bei Bedarf ein Foto gemacht.
    """

    def __init__(self):
        self.camera = None
        self.camera_lock = threading.Lock()
        self.pipe = None
        self.stats = {'snapshots': 0, 'detections': 0}
        self._analyzing = False

        # Vision-Pipeline laden
        print("\n🔧 Lade Vision-Modelle...")
        self.pipe = TwoStageONNX(
            det_onnx='yolo11n.onnx',
            cls_onnx='mobilenet_v3_small_253cls.onnx',
            classes_txt='classes.txt',
            config_json='vision_pipeline_config.json',
            threads=4
        )
        print(f"✅ Vision-Pipeline bereit!")

        # Kamera initialisieren
        self._init_camera()

    def _init_camera(self):
        """Kamera initialisieren (Picamera2 oder OpenCV-Fallback)."""
        print("📷 Initialisiere Kamera...")
        if PICAMERA_AVAILABLE:
            self.camera = Picamera2()
            cam_config = self.camera.create_still_configuration(
                main={"size": (VISION_WIDTH, VISION_HEIGHT), "format": "RGB888"}
            )
            self.camera.configure(cam_config)
            self.camera.start()
            time.sleep(1.0)  # Warmup
            print(f"✅ Picamera2 bereit ({VISION_WIDTH}x{VISION_HEIGHT})")
        else:
            self.camera = cv2.VideoCapture(0)
            if self.camera.isOpened():
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, VISION_WIDTH)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, VISION_HEIGHT)
                print(f"✅ OpenCV-Kamera bereit")
            else:
                print("❌ Keine Kamera verfügbar!")
                self.camera = None

    def capture_snapshot(self) -> Optional[np.ndarray]:
        """Ein einzelnes Bild aufnehmen. Gibt RGB-Array zurück."""
        with self.camera_lock:
            if self.camera is None:
                return None
            if PICAMERA_AVAILABLE:
                return self.camera.capture_array()
            else:
                # OpenCV-Fallback: Buffer leeren + lesen
                self.camera.grab()
                self.camera.grab()
                ok, frame_bgr = self.camera.read()
                if ok:
                    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                return None

    def analyze_snapshot(self) -> Dict[str, Any]:
        """
        Snapshot aufnehmen → Two-Stage-Pipeline → Ergebnis.
        Wird per /api/camera/capture aufgerufen.
        """
        if self._analyzing:
            return {"success": False, "error": "Analyse läuft bereits"}

        self._analyzing = True
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        try:
            # 1) Bild aufnehmen
            frame_rgb = self.capture_snapshot()
            if frame_rgb is None:
                return {"success": False, "error": "Kamera nicht verfügbar", "timestamp": timestamp}

            self.stats['snapshots'] += 1

            # 2) Pipeline ausführen
            t0 = time.time()
            result = self.pipe.predict(frame_rgb)
            inference_ms = (time.time() - t0) * 1000

            # 3) Annotiertes Bild erstellen
            annotated_rgb = draw_result(frame_rgb, result)
            annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

            # Thumbnail (kleiner für Historie)
            thumb_h = 180
            h, w = annotated_bgr.shape[:2]
            thumb_w = int(w * (thumb_h / h))
            thumb = cv2.resize(annotated_bgr, (thumb_w, thumb_h))
            _, thumb_jpeg = cv2.imencode('.jpg', thumb, [cv2.IMWRITE_JPEG_QUALITY, 75])
            thumb_b64 = base64.b64encode(thumb_jpeg.tobytes()).decode('utf-8')

            # Vollbild
            _, full_jpeg = cv2.imencode('.jpg', annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            full_b64 = base64.b64encode(full_jpeg.tobytes()).decode('utf-8')

            # Original (ohne Annotationen)
            orig_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            _, orig_jpeg = cv2.imencode('.jpg', orig_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            orig_b64 = base64.b64encode(orig_jpeg.tobytes()).decode('utf-8')

            # 4) Ergebnis zusammenbauen
            detections = result.get('detections', [])
            winner = result.get('winner')

            # Deutsche Namen mappen
            for det in detections:
                sci = det.get('species_name', '')
                det['species_german'] = get_german_name(sci)

            if winner:
                winner['species_german'] = get_german_name(winner.get('species_name', ''))
                self.stats['detections'] += 1

            return {
                "success": True,
                "timestamp": timestamp,
                "detections": detections,
                "winner": winner,
                "annotated_image_b64": full_b64,
                "original_image_b64": orig_b64,
                "thumbnail_b64": thumb_b64,
                "inference_time_ms": round(inference_ms, 1),
            }

        except Exception as e:
            print(f"❌ Vision-Fehler: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e), "timestamp": timestamp}
        finally:
            self._analyzing = False

    def get_preview_jpeg(self) -> Optional[bytes]:
        """Einzelnes Preview-Bild als JPEG (für Kamera-Vorschau)."""
        frame_rgb = self.capture_snapshot()
        if frame_rgb is None:
            return None
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        # Verkleinern für schnelle Übertragung
        h, w = frame_bgr.shape[:2]
        scale = 640 / w
        small = cv2.resize(frame_bgr, (640, int(h * scale)))
        _, jpeg = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 60])
        return jpeg.tobytes()

    def stop(self):
        with self.camera_lock:
            if self.camera is not None:
                if PICAMERA_AVAILABLE:
                    self.camera.stop()
                else:
                    self.camera.release()
                self.camera = None


# ============================================================================
# FLASK APP
# ============================================================================

app = Flask(__name__)
app.config['SECRET_KEY'] = 'zwitscherkasten-secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# Globale Variablen
audio_processor = None
vision_processor = None
detections = []
stats = {
    'start_time': None,
    'species_today': set(),
    'detections_today': 0,
}


def on_detection(detection):
    """Callback für neue Erkennung (Audio oder Vision)."""
    global detections, stats
    detections.insert(0, detection)
    if len(detections) > MAX_HISTORY:
        detections.pop()
    stats['species_today'].add(detection.species)
    stats['detections_today'] += 1


# ── Statische Dateien ───────────────────────────────────────

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(os.path.join(os.path.dirname(__file__), 'static'), filename)


# ── API-Endpunkte ──────────────────────────────────────────

@app.route('/api/detections')
def get_detections():
    return jsonify([d.to_dict() for d in detections])


@app.route('/api/stats')
def get_stats():
    uptime = ""
    if stats['start_time']:
        delta = datetime.now() - stats['start_time']
        hours, remainder = divmod(int(delta.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        uptime = f"{hours}h {minutes}min {seconds}s"

    return jsonify({
        'uptime': uptime,
        'audio_triggers': audio_processor.stats['intent_triggers'] if audio_processor else 0,
        'audio_chunks': audio_processor.stats['total_chunks'] if audio_processor else 0,
        'vision_snapshots': vision_processor.stats['snapshots'] if vision_processor else 0,
        'vision_detections': vision_processor.stats['detections'] if vision_processor else 0,
        'detections_today': stats['detections_today'],
        'species_today': len(stats['species_today']),
        'audio_level': audio_processor.audio_level if audio_processor else 0,
        'camera_available': vision_processor is not None and vision_processor.camera is not None,
    })


@app.route('/api/camera/preview')
def camera_preview():
    """Einzelnes Vorschaubild als JPEG."""
    if not vision_processor:
        return jsonify({"error": "Vision nicht aktiv"}), 503
    jpeg = vision_processor.get_preview_jpeg()
    if jpeg is None:
        return jsonify({"error": "Kein Bild verfügbar"}), 503
    return Response(jpeg, mimetype='image/jpeg')


@app.route('/api/camera/capture', methods=['POST'])
def camera_capture():
    """Snapshot aufnehmen und analysieren."""
    if not vision_processor:
        return jsonify({"success": False, "error": "Vision nicht aktiv"}), 503

    result = vision_processor.analyze_snapshot()

    # Bei Erfolg: Detection in Historie einfügen
    if result['success'] and result.get('winner'):
        w = result['winner']
        species_name = w.get('species_german', w.get('species_name', 'Unbekannt'))
        species_sci = w.get('species_name', '').replace('_', ' ')
        detection = Detection(
            timestamp=result['timestamp'],
            species=species_name,
            species_scientific=species_sci,
            confidence=float(w.get('score', 0.0)),
            source="vision",
            thumbnail_b64=result.get('thumbnail_b64', '')
        )
        print(f"📷 {detection.timestamp} | {species_name} | {detection.confidence:.0%}")
        on_detection(detection)

    return jsonify(result)


# ── HTML-Template ──────────────────────────────────────────

HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Zwitscherkasten</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,500;0,9..40,700;0,9..40,800;1,9..40,400&display=swap');

:root {
    --brand: #0D5961;
    --brand-light: #10707A;
    --accent: #FF7359;
    --accent-glow: rgba(255, 115, 89, 0.15);
    --bg: #F2F5F3;
    --text: #1A2626;
    --gray: #667373;
    --gray-light: #9BAAAA;
    --white: #FFFFFF;
    --audio: #0D5961;
    --vision: #8B5CF6;
    --vision-light: rgba(139, 92, 246, 0.1);
    --radius: 18px;
    --radius-sm: 12px;
    --shadow: 0 1px 3px rgba(0,0,0,0.04), 0 4px 12px rgba(0,0,0,0.06);
    --shadow-lg: 0 4px 20px rgba(0,0,0,0.1);
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    background: var(--bg);
    min-height: 100vh;
    color: var(--text);
    padding: 16px;
    -webkit-font-smoothing: antialiased;
}
.container { max-width: 1120px; margin: 0 auto; }

/* ── Header ─────────────────────────────── */
header { text-align: center; padding: 20px 0 16px; }
.logo-row { display: flex; align-items: center; justify-content: center; gap: 14px; margin-bottom: 4px; }
.logo { width: 48px; height: 48px; }
header h1 { font-size: 1.9em; font-weight: 800; color: var(--text); letter-spacing: -0.5px; }
header p { color: var(--gray); font-size: 0.85em; font-weight: 500; }

/* ── Main Grid ──────────────────────────── */
.main-grid {
    display: grid;
    grid-template-columns: 1fr 380px;
    gap: 16px;
    margin-bottom: 16px;
}

/* ── Kamera-Karte ───────────────────────── */
.camera-card {
    background: var(--white);
    border-radius: var(--radius);
    overflow: hidden;
    box-shadow: var(--shadow);
}
.camera-header {
    background: var(--vision);
    color: var(--white);
    padding: 12px 18px;
    font-weight: 700;
    font-size: 0.85em;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.camera-header .badge {
    font-size: 0.75em;
    background: rgba(255,255,255,0.2);
    padding: 3px 10px;
    border-radius: 20px;
    font-weight: 500;
}
.camera-body {
    position: relative;
    background: #0a0a0a;
    min-height: 300px;
    display: flex;
    align-items: center;
    justify-content: center;
}
.camera-body img {
    width: 100%;
    display: block;
}
.camera-placeholder {
    color: var(--gray-light);
    text-align: center;
    padding: 60px 20px;
}
.camera-placeholder .cam-icon { font-size: 3em; margin-bottom: 12px; opacity: 0.4; }
.camera-placeholder p { font-size: 0.85em; }

.camera-actions {
    padding: 14px 18px;
    display: flex;
    gap: 10px;
    align-items: center;
    border-top: 1px solid var(--bg);
}
.btn {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 10px 22px;
    border: none;
    border-radius: var(--radius-sm);
    font-family: inherit;
    font-size: 0.85em;
    font-weight: 700;
    cursor: pointer;
    transition: all 0.15s ease;
}
.btn:active { transform: scale(0.97); }
.btn-capture {
    background: var(--accent);
    color: var(--white);
    flex: 1;
}
.btn-capture:hover { background: #e8613f; box-shadow: 0 4px 15px var(--accent-glow); }
.btn-capture:disabled {
    background: var(--gray-light);
    cursor: not-allowed;
    box-shadow: none;
}
.btn-capture:disabled:active { transform: none; }
.btn-preview {
    background: var(--bg);
    color: var(--text);
    padding: 10px 16px;
}
.btn-preview:hover { background: #e3e8e4; }

.capture-result {
    padding: 14px 18px;
    border-top: 1px solid var(--bg);
    display: none;
}
.capture-result.visible { display: block; }
.result-winner {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 12px 16px;
    background: var(--bg);
    border-radius: var(--radius-sm);
}
.result-winner .species-icon {
    width: 44px;
    height: 44px;
    background: var(--vision-light);
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5em;
}
.result-winner .species-info { flex: 1; }
.result-winner .species-name { font-weight: 700; font-size: 1.05em; }
.result-winner .species-sci { font-size: 0.75em; color: var(--gray); font-style: italic; }
.result-winner .score {
    font-weight: 800;
    font-size: 1.2em;
    color: var(--vision);
}
.result-meta {
    display: flex;
    gap: 16px;
    margin-top: 8px;
    font-size: 0.75em;
    color: var(--gray);
}
.result-none {
    text-align: center;
    padding: 12px;
    color: var(--gray);
    font-size: 0.85em;
}

/* ── Analyzing Overlay ──────────────────── */
.analyzing-overlay {
    position: absolute;
    inset: 0;
    background: rgba(0,0,0,0.6);
    display: none;
    align-items: center;
    justify-content: center;
    flex-direction: column;
    gap: 12px;
    z-index: 10;
}
.analyzing-overlay.visible { display: flex; }
.spinner {
    width: 40px;
    height: 40px;
    border: 3px solid rgba(255,255,255,0.2);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
.analyzing-overlay span { color: #fff; font-size: 0.85em; font-weight: 600; }

/* ── Sidebar ────────────────────────────── */
.sidebar { display: flex; flex-direction: column; gap: 14px; }

.status-card, .audio-card {
    background: var(--white);
    border-radius: var(--radius);
    padding: 16px;
    box-shadow: var(--shadow);
}
.card-title {
    font-size: 0.72em;
    color: var(--gray);
    margin-bottom: 10px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 700;
}
.status-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.85em;
    padding: 5px 0;
    border-bottom: 1px solid var(--bg);
}
.status-row:last-child { border: none; }
.status-row .label { color: var(--gray); }
.status-row .value { font-weight: 700; }
.status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 6px;
    animation: pulse 2s ease infinite;
}
.status-dot.active { background: #10b981; }
.status-dot.inactive { background: var(--gray-light); animation: none; }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }

.audio-bars { display: flex; align-items: flex-end; gap: 2px; height: 28px; }
.audio-bar {
    flex: 1;
    background: #e8edea;
    border-radius: 2px;
    min-height: 2px;
    transition: height 0.1s ease;
}
.audio-bar.active { background: var(--audio); }

/* ── Stats Row ──────────────────────────── */
.stats-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin-bottom: 14px;
}
.stat-mini {
    background: var(--white);
    border-radius: 14px;
    padding: 14px 10px;
    text-align: center;
    box-shadow: var(--shadow);
}
.stat-mini .number { font-size: 1.5em; font-weight: 800; color: var(--brand); }
.stat-mini .label { font-size: 0.62em; color: var(--gray); margin-top: 2px; }

/* ── Detections-Liste ───────────────────── */
.detections-container {
    background: var(--white);
    border-radius: var(--radius);
    overflow: hidden;
    box-shadow: var(--shadow);
}
.detections-header {
    background: var(--brand);
    color: var(--white);
    padding: 12px 18px;
    font-weight: 700;
    font-size: 0.85em;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.detections-list { max-height: 400px; overflow-y: auto; }
.detection {
    padding: 12px 18px;
    border-bottom: 1px solid var(--bg);
    display: flex;
    align-items: center;
    gap: 12px;
    transition: background 0.1s;
}
.detection:hover { background: var(--bg); }

.detection-thumb {
    width: 52px;
    height: 38px;
    border-radius: 8px;
    overflow: hidden;
    background: var(--bg);
    flex-shrink: 0;
}
.detection-thumb img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}
.detection-icon {
    width: 38px;
    height: 38px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1em;
    flex-shrink: 0;
}
.detection-icon.audio { background: rgba(13,89,97,0.1); }
.detection-icon.vision { background: var(--vision-light); }
.detection-info { flex: 1; min-width: 0; }
.detection-species { font-weight: 700; font-size: 0.88em; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.detection-scientific { font-size: 0.68em; color: var(--gray); font-style: italic; }
.detection-meta { text-align: right; flex-shrink: 0; }
.detection-time { font-size: 0.72em; color: var(--gray); }
.detection-conf { font-size: 0.82em; font-weight: 700; }
.detection-conf.audio { color: var(--audio); }
.detection-conf.vision { color: var(--vision); }

.empty-state { text-align: center; padding: 40px 20px; color: var(--gray); }
.empty-state .icon { font-size: 2.5em; opacity: 0.3; margin-bottom: 10px; }
.empty-state p { font-size: 0.85em; }

/* ── Footer ─────────────────────────────── */
footer { text-align: center; padding: 16px; color: var(--gray-light); font-size: 0.72em; }

/* ── Responsive ─────────────────────────── */
@media (max-width: 860px) {
    .main-grid { grid-template-columns: 1fr; }
    .stats-row { grid-template-columns: repeat(2, 1fr); }
}
</style>
</head>
<body>
<div class="container">

<header>
    <div class="logo-row">
        <img src="/static/logo.svg" alt="Logo" class="logo">
        <h1>Zwitscherkasten</h1>
    </div>
    <p>Audio + Kamera Vogelerkennung</p>
</header>

<div class="main-grid">

    <!-- Kamera-Karte -->
    <div class="camera-card">
        <div class="camera-header">
            <span>📷 Kamera</span>
            <span class="badge" id="cameraStatus">Bereit</span>
        </div>
        <div class="camera-body">
            <div class="analyzing-overlay" id="analyzeOverlay">
                <div class="spinner"></div>
                <span>Analysiere …</span>
            </div>
            <div class="camera-placeholder" id="cameraPlaceholder">
                <div class="cam-icon">📷</div>
                <p>Klicke auf <b>Vorschau</b> um die Kamera zu sehen,<br>dann auf <b>Foto &amp; Erkennen</b> um ein Bild zu analysieren.</p>
            </div>
            <img id="cameraImage" style="display:none" alt="Kamera">
        </div>
        <div class="camera-actions">
            <button class="btn btn-capture" id="captureBtn" onclick="captureAndAnalyze()">
                📸 Foto &amp; Erkennen
            </button>
            <button class="btn btn-preview" id="previewBtn" onclick="loadPreview()">
                👁 Vorschau
            </button>
        </div>
        <div class="capture-result" id="captureResult">
            <!-- wird dynamisch befüllt -->
        </div>
    </div>

    <!-- Sidebar -->
    <div class="sidebar">
        <div class="status-card">
            <div class="card-title">System Status</div>
            <div class="status-row">
                <span class="label">Laufzeit</span>
                <span class="value" id="uptime">--</span>
            </div>
            <div class="status-row">
                <span class="label">Audio</span>
                <span class="value"><span class="status-dot active" id="audioDot"></span><span id="audioStatus">Aktiv</span></span>
            </div>
            <div class="status-row">
                <span class="label">Kamera</span>
                <span class="value"><span class="status-dot active" id="cameraDot"></span><span id="cameraStatusText">Bereit</span></span>
            </div>
            <div class="status-row">
                <span class="label">Snapshots</span>
                <span class="value" id="visionSnapshots">0</span>
            </div>
        </div>

        <div class="audio-card">
            <div class="card-title">🎤 Mikrofon-Pegel</div>
            <div class="audio-bars" id="audioBars"></div>
        </div>

        <div class="detections-container">
            <div class="detections-header">
                <span>Erkennungen</span>
                <span id="detectionCount">0</span>
            </div>
            <div class="detections-list" id="detectionsList">
                <div class="empty-state">
                    <div class="icon">🐦</div>
                    <p>Warte auf Erkennungen …</p>
                </div>
            </div>
        </div>
    </div>
</div>

<div class="stats-row">
    <div class="stat-mini"><div class="number" id="detectionsToday">0</div><div class="label">Erkennungen</div></div>
    <div class="stat-mini"><div class="number" id="speciesToday">0</div><div class="label">Arten</div></div>
    <div class="stat-mini"><div class="number" id="audioTriggers">0</div><div class="label">Audio</div></div>
    <div class="stat-mini"><div class="number" id="visionDetCount">0</div><div class="label">Kamera</div></div>
</div>

<footer>Zwitscherkasten &middot; TH Ingolstadt &middot; Applied AI</footer>
</div>

<script>
var detections = [];
var barCount = 30;
var barsHtml = "";
for (var i = 0; i < barCount; i++) barsHtml += "<div class='audio-bar'></div>";
document.getElementById("audioBars").innerHTML = barsHtml;

/* ── Vorschau laden ──────────────────── */
function loadPreview() {
    var img = document.getElementById("cameraImage");
    var ph = document.getElementById("cameraPlaceholder");
    img.src = "/api/camera/preview?" + Date.now();
    img.style.display = "block";
    ph.style.display = "none";
    img.onerror = function() {
        ph.style.display = "block";
        ph.innerHTML = "<div class='cam-icon'>⚠️</div><p>Kamera nicht erreichbar</p>";
        img.style.display = "none";
    };
}

/* ── Foto + Analyse ──────────────────── */
function captureAndAnalyze() {
    var btn = document.getElementById("captureBtn");
    var overlay = document.getElementById("analyzeOverlay");
    var statusBadge = document.getElementById("cameraStatus");
    var resultDiv = document.getElementById("captureResult");

    btn.disabled = true;
    overlay.classList.add("visible");
    statusBadge.textContent = "Analysiert …";
    resultDiv.classList.remove("visible");

    var xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/camera/capture", true);
    xhr.onreadystatechange = function() {
        if (xhr.readyState !== 4) return;
        btn.disabled = false;
        overlay.classList.remove("visible");
        statusBadge.textContent = "Bereit";

        if (xhr.status !== 200) {
            resultDiv.innerHTML = "<div class='result-none'>❌ Fehler bei der Analyse</div>";
            resultDiv.classList.add("visible");
            return;
        }

        var data = JSON.parse(xhr.responseText);
        if (!data.success) {
            resultDiv.innerHTML = "<div class='result-none'>⚠️ " + (data.error || "Fehler") + "</div>";
            resultDiv.classList.add("visible");
            return;
        }

        // Annotiertes Bild anzeigen
        var img = document.getElementById("cameraImage");
        var ph = document.getElementById("cameraPlaceholder");
        img.src = "data:image/jpeg;base64," + data.annotated_image_b64;
        img.style.display = "block";
        ph.style.display = "none";

        // Ergebnis anzeigen
        if (data.winner) {
            var name = data.winner.species_german || data.winner.species_name || "Unbekannt";
            var sci = (data.winner.species_name || "").replace(/_/g, " ");
            var score = Math.round((data.winner.score || 0) * 100);
            resultDiv.innerHTML =
                "<div class='result-winner'>" +
                "  <div class='species-icon'>🐦</div>" +
                "  <div class='species-info'>" +
                "    <div class='species-name'>" + name + "</div>" +
                "    <div class='species-sci'>" + sci + "</div>" +
                "  </div>" +
                "  <div class='score'>" + score + "%</div>" +
                "</div>" +
                "<div class='result-meta'>" +
                "  <span>⏱ " + data.inference_time_ms + " ms</span>" +
                "  <span>🎯 " + data.detections.length + " Erkennung(en)</span>" +
                "</div>";
        } else {
            resultDiv.innerHTML = "<div class='result-none'>Kein Vogel erkannt – versuche es erneut wenn ein Vogel sichtbar ist.</div>";
        }
        resultDiv.classList.add("visible");

        // Stats sofort aktualisieren
        loadStats();
        loadDetections();
    };
    xhr.send();
}

/* ── Polling ─────────────────────────── */
function loadStats() {
    var x = new XMLHttpRequest();
    x.open("GET", "/api/stats", true);
    x.onreadystatechange = function() {
        if (x.readyState !== 4 || x.status !== 200) return;
        var s = JSON.parse(x.responseText);
        document.getElementById("uptime").textContent = s.uptime || "--";
        document.getElementById("detectionsToday").textContent = s.detections_today || 0;
        document.getElementById("speciesToday").textContent = s.species_today || 0;
        document.getElementById("audioTriggers").textContent = s.audio_triggers || 0;
        document.getElementById("visionSnapshots").textContent = s.vision_snapshots || 0;
        document.getElementById("visionDetCount").textContent = s.vision_detections || 0;
        updateAudioMeter(s.audio_level || 0);

        // Kamera-Status
        var camDot = document.getElementById("cameraDot");
        var camText = document.getElementById("cameraStatusText");
        if (s.camera_available) {
            camDot.className = "status-dot active";
            camText.textContent = "Bereit";
        } else {
            camDot.className = "status-dot inactive";
            camText.textContent = "Offline";
        }
    };
    x.send();
}

function loadDetections() {
    var x = new XMLHttpRequest();
    x.open("GET", "/api/detections", true);
    x.onreadystatechange = function() {
        if (x.readyState !== 4 || x.status !== 200) return;
        detections = JSON.parse(x.responseText);
        renderDetections();
    };
    x.send();
}

function updateAudioMeter(level) {
    var bars = document.querySelectorAll(".audio-bar");
    var activeBars = Math.floor((level / 100) * barCount);
    for (var i = 0; i < bars.length; i++) {
        if (i < activeBars) {
            bars[i].style.height = (4 + Math.random() * 20) + "px";
            bars[i].classList.add("active");
        } else {
            bars[i].style.height = "2px";
            bars[i].classList.remove("active");
        }
    }
}

function renderDetections() {
    var list = document.getElementById("detectionsList");
    document.getElementById("detectionCount").textContent = detections.length;
    if (!detections.length) {
        list.innerHTML = "<div class='empty-state'><div class='icon'>🐦</div><p>Warte auf Erkennungen …</p></div>";
        return;
    }
    var html = "";
    var max = Math.min(detections.length, 20);
    for (var i = 0; i < max; i++) {
        var d = detections[i];
        var conf = Math.round(d.confidence * 100);
        var t = d.timestamp.split(" ")[1];
        var isVision = d.source === "vision";
        var icon = isVision ? "📷" : "🎤";

        html += "<div class='detection'>";

        // Thumbnail oder Icon
        if (isVision && d.thumbnail_b64) {
            html += "<div class='detection-thumb'><img src='data:image/jpeg;base64," + d.thumbnail_b64 + "'></div>";
        } else {
            html += "<div class='detection-icon " + d.source + "'>" + icon + "</div>";
        }

        html += "<div class='detection-info'>";
        html += "<div class='detection-species'>" + d.species + "</div>";
        html += "<div class='detection-scientific'>" + d.species_scientific + "</div>";
        html += "</div>";
        html += "<div class='detection-meta'>";
        html += "<div class='detection-time'>" + t + "</div>";
        html += "<div class='detection-conf " + d.source + "'>" + conf + "%</div>";
        html += "</div></div>";
    }
    list.innerHTML = html;
}

loadStats();
loadDetections();
setInterval(loadStats, 2000);
setInterval(loadDetections, 3000);
</script>
</body>
</html>'''


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


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
    global audio_processor, vision_processor, stats

    parser = argparse.ArgumentParser(description="🐦 Zwitscherkasten Live (Audio + Snapshot-Vision)")
    parser.add_argument('--port', '-p', type=int, default=5000)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--device', '-d', type=int, default=None, help='Audio-Gerät')
    parser.add_argument('--list-devices', '-l', action='store_true')
    parser.add_argument('--no-audio', action='store_true', help='Ohne Audio')
    parser.add_argument('--no-vision', action='store_true', help='Ohne Kamera')
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    print("\n" + "=" * 60)
    print("🐦 ZWITSCHERKASTEN – Audio + Snapshot-Vision")
    print("=" * 60)

    stats['start_time'] = datetime.now()

    # Vision initialisieren (Snapshot-Modus)
    if not args.no_vision:
        try:
            vision_processor = VisionProcessor()
        except Exception as e:
            print(f"⚠️ Vision konnte nicht gestartet werden: {e}")
            vision_processor = None
    else:
        print("⚠️ Vision deaktiviert")

    # Audio starten
    if not args.no_audio:
        audio_processor = AudioProcessor(on_detection, device=args.device)
        audio_processor.start()
    else:
        print("⚠️ Audio deaktiviert")

    print(f"\n🌐 http://localhost:{args.port}")
    print(f"   Drücke Ctrl+C zum Beenden\n")

    try:
        socketio.run(app, host=args.host, port=args.port, debug=False, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        pass
    finally:
        if audio_processor:
            audio_processor.stop()
        if vision_processor:
            vision_processor.stop()
        print("\n👋 Auf Wiedersehen!\n")


if __name__ == "__main__":
    main()