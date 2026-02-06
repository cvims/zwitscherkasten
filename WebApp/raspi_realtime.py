#!/usr/bin/env python3
"""
🐦 Zwitscherkasten - Raspberry Pi Echtzeit-Erkennung
====================================================

Dieses Skript läuft kontinuierlich und erkennt Vogelstimmen
in Echtzeit über das angeschlossene Mikrofon.

Voraussetzungen:
    pip install sounddevice numpy librosa onnxruntime tflite-runtime

Nutzung:
    python raspi_realtime.py                    # Standard-Mikrofon
    python raspi_realtime.py --device 1         # Spezifisches Gerät
    python raspi_realtime.py --list-devices     # Verfügbare Geräte zeigen
    python raspi_realtime.py --threshold 0.7    # Höhere Schwelle
"""

import argparse
import time
import json
import queue
import threading
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import deque

# Versuche sounddevice zu importieren
try:
    import sounddevice as sd
except ImportError:
    print("❌ sounddevice nicht installiert!")
    print("   pip install sounddevice")
    exit(1)

from zwitscherkasten_inference import Zwitscherkasten, Config


# ============================================================================
# KONFIGURATION
# ============================================================================

class RealtimeConfig:
    """Konfiguration für Echtzeit-Erkennung."""
    
    # Audio-Aufnahme
    sample_rate: int = 22050         # Sample Rate
    chunk_duration: float = 3.0      # Sekunden pro Analyse-Chunk
    overlap: float = 0.5             # Überlappung zwischen Chunks (50%)
    
    # Erkennung
    intent_threshold: float = 0.5    # Schwelle für Vogelerkennung
    species_threshold: float = 0.3   # Mindest-Konfidenz für Artenausgabe
    cooldown_seconds: float = 2.0    # Mindestzeit zwischen Meldungen derselben Art
    
    # Logging
    log_to_file: bool = True
    log_dir: str = "detections"


# ============================================================================
# ECHTZEIT-ERKENNUNGS-MANAGER
# ============================================================================

class RealtimeDetector:
    """Verwaltet die Echtzeit-Vogelerkennung."""
    
    def __init__(self, config: RealtimeConfig, device: int = None):
        self.config = config
        self.device = device
        
        # Audio-Puffer
        self.chunk_samples = int(config.sample_rate * config.chunk_duration)
        self.hop_samples = int(self.chunk_samples * (1 - config.overlap))
        self.audio_buffer = deque(maxlen=self.chunk_samples * 2)
        
        # Verarbeitungs-Queue
        self.audio_queue = queue.Queue()
        self.running = False
        
        # Cooldown für wiederholte Erkennungen
        self.last_detection = {}  # species -> timestamp
        
        # Statistiken
        self.stats = {
            'total_chunks': 0,
            'bird_detections': 0,
            'species_detections': {},
            'start_time': None
        }
        
        # Zwitscherkasten laden
        print("\n🔧 Initialisiere Zwitscherkasten...")
        zk_config = Config()
        zk_config.intent_threshold = config.intent_threshold
        zk_config.top_k = 3
        self.zk = Zwitscherkasten(zk_config)
        
        # Log-Verzeichnis
        if config.log_to_file:
            Path(config.log_dir).mkdir(exist_ok=True)
            self.log_file = Path(config.log_dir) / f"detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        
        print(f"\n✅ Echtzeit-Erkennung bereit!")
        print(f"   Chunk-Dauer: {config.chunk_duration}s")
        print(f"   Überlappung: {config.overlap*100:.0f}%")
        print(f"   Schwellen: Intent={config.intent_threshold}, Species={config.species_threshold}")
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Callback für Audio-Stream."""
        if status:
            print(f"⚠️  Audio-Status: {status}")
        
        # Audio in Puffer
        audio = indata[:, 0].copy()  # Mono
        self.audio_buffer.extend(audio)
        
        # Wenn genug Daten, zur Verarbeitung schicken
        if len(self.audio_buffer) >= self.chunk_samples:
            chunk = np.array(list(self.audio_buffer)[:self.chunk_samples])
            # Entferne verarbeitete Samples (mit Überlappung)
            for _ in range(self.hop_samples):
                if self.audio_buffer:
                    self.audio_buffer.popleft()
            
            try:
                self.audio_queue.put_nowait(chunk)
            except queue.Full:
                pass  # Überspringe wenn Verarbeitung zu langsam
    
    def _process_loop(self):
        """Verarbeitungs-Thread."""
        while self.running:
            try:
                chunk = self.audio_queue.get(timeout=1.0)
                self._process_chunk(chunk)
            except queue.Empty:
                continue
    
    def _process_chunk(self, audio: np.ndarray):
        """Verarbeitet einen Audio-Chunk."""
        self.stats['total_chunks'] += 1
        
        # Analyse mit Zwitscherkasten
        start = time.perf_counter()
        result = self.zk.analyze_stream(audio, self.config.sample_rate)
        process_time = (time.perf_counter() - start) * 1000
        
        # Kein Vogel erkannt
        if not result['bird_detected']:
            return
        
        self.stats['bird_detections'] += 1
        
        # Vogelart prüfen
        if result['species'] and result['species']['confidence'] >= self.config.species_threshold:
            species = result['species']['species']
            confidence = result['species']['confidence']
            
            # Cooldown prüfen
            now = time.time()
            if species in self.last_detection:
                elapsed = now - self.last_detection[species]
                if elapsed < self.config.cooldown_seconds:
                    return
            
            self.last_detection[species] = now
            
            # Statistik aktualisieren
            if species not in self.stats['species_detections']:
                self.stats['species_detections'][species] = 0
            self.stats['species_detections'][species] += 1
            
            # Ausgabe
            timestamp = datetime.now().strftime('%H:%M:%S')
            print(f"\n🐦 [{timestamp}] {species}")
            print(f"   Konfidenz: {confidence:.1%} | Intent: {result['intent_confidence']:.1%}")
            print(f"   Verarbeitung: {process_time:.0f}ms")
            
            # Top-3 zeigen
            if len(result['predictions']) > 1:
                others = [f"{p['species']} ({p['confidence']:.0%})" 
                         for p in result['predictions'][1:3]]
                print(f"   Alternativen: {', '.join(others)}")
            
            # Logging
            if self.config.log_to_file:
                log_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'species': species,
                    'species_scientific': result['species']['species_scientific'],
                    'confidence': confidence,
                    'intent_confidence': result['intent_confidence'],
                    'processing_time_ms': process_time,
                    'predictions': result['predictions']
                }
                with open(self.log_file, 'a') as f:
                    f.write(json.dumps(log_entry) + '\n')
    
    def run(self):
        """Startet die Echtzeit-Erkennung."""
        self.running = True
        self.stats['start_time'] = datetime.now()
        
        # Verarbeitungs-Thread starten
        process_thread = threading.Thread(target=self._process_loop, daemon=True)
        process_thread.start()
        
        print("\n" + "="*60)
        print("🎤 ECHTZEIT-ERKENNUNG GESTARTET")
        print("="*60)
        print(f"   Drücke Ctrl+C zum Beenden")
        print("="*60 + "\n")
        
        try:
            with sd.InputStream(
                device=self.device,
                channels=1,
                samplerate=self.config.sample_rate,
                blocksize=int(self.config.sample_rate * 0.1),  # 100ms Blöcke
                callback=self._audio_callback
            ):
                while self.running:
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            print("\n\n⏹️  Beende Erkennung...")
        finally:
            self.running = False
            time.sleep(0.5)  # Thread beenden lassen
            self._print_stats()
    
    def _print_stats(self):
        """Zeigt Statistiken an."""
        if self.stats['start_time']:
            duration = datetime.now() - self.stats['start_time']
            minutes = duration.total_seconds() / 60
            
            print("\n" + "="*60)
            print("📊 STATISTIKEN")
            print("="*60)
            print(f"   Laufzeit: {minutes:.1f} Minuten")
            print(f"   Analysierte Chunks: {self.stats['total_chunks']}")
            print(f"   Vogel-Erkennungen: {self.stats['bird_detections']}")
            
            if self.stats['species_detections']:
                print(f"\n   Erkannte Arten:")
                sorted_species = sorted(
                    self.stats['species_detections'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                for species, count in sorted_species[:10]:
                    print(f"      • {species}: {count}x")
            
            if self.config.log_to_file:
                print(f"\n   Log-Datei: {self.log_file}")
            
            print("="*60 + "\n")


# ============================================================================
# CLI
# ============================================================================

def list_devices():
    """Zeigt verfügbare Audio-Geräte."""
    print("\n🎤 Verfügbare Audio-Geräte:")
    print("-" * 50)
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            star = " ⭐" if i == sd.default.device[0] else ""
            print(f"   [{i}] {dev['name']}{star}")
            print(f"       Kanäle: {dev['max_input_channels']}, "
                  f"Sample Rate: {int(dev['default_samplerate'])} Hz")
    print("-" * 50)
    print("   ⭐ = Standard-Eingabegerät")


def main():
    parser = argparse.ArgumentParser(
        description="🐦 Zwitscherkasten Echtzeit-Erkennung",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--device', '-d', type=int, default=None,
                       help='Audio-Eingabegerät (Index)')
    parser.add_argument('--list-devices', '-l', action='store_true',
                       help='Verfügbare Geräte anzeigen')
    parser.add_argument('--threshold', '-t', type=float, default=0.5,
                       help='Intent-Schwelle (0.0-1.0, Standard: 0.5)')
    parser.add_argument('--species-threshold', '-s', type=float, default=0.3,
                       help='Arten-Konfidenz-Schwelle (Standard: 0.3)')
    parser.add_argument('--chunk-duration', '-c', type=float, default=3.0,
                       help='Chunk-Dauer in Sekunden (Standard: 3.0)')
    parser.add_argument('--no-log', action='store_true',
                       help='Logging deaktivieren')
    
    args = parser.parse_args()
    
    if args.list_devices:
        list_devices()
        return
    
    # Konfiguration
    config = RealtimeConfig()
    config.intent_threshold = args.threshold
    config.species_threshold = args.species_threshold
    config.chunk_duration = args.chunk_duration
    config.log_to_file = not args.no_log
    
    # Erkennung starten
    detector = RealtimeDetector(config, device=args.device)
    detector.run()


if __name__ == "__main__":
    main()
