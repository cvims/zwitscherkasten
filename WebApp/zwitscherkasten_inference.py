"""
🐦 Zwitscherkasten - Zweistufige Vogelerkennungs-Pipeline
=========================================================

Stufe 1: Intent-Modell (TFLite) - "Ist da ein Vogel?" (schnell, ~12KB)
Stufe 2: EfficientNet-B0 (ONNX) - "Welche Art?" (genau, ~26MB)

Für Raspberry Pi optimiert mit optionaler GPU-Beschleunigung.
"""

import os
import json
import time
import numpy as np
import librosa
from pathlib import Path
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass


# ============================================================================
# KONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Zentrale Konfiguration für beide Modelle."""
    
    # Intent-Modell (Stufe 1)
    intent_model_path: str = "bird_intent_model.tflite"
    intent_sample_rate: int = 16000
    intent_duration: float = 2.0
    intent_n_mels: int = 64
    intent_hop_length: int = 512
    intent_threshold: float = 0.5  # Schwellwert für "Vogel erkannt"
    
    # EfficientNet (Stufe 2) - Parameter müssen mit Training übereinstimmen!
    efficientnet_model_path: str = "efficientnet_b0_audio.onnx"
    efficientnet_sample_rate: int = 32000  # Training: 32kHz
    efficientnet_duration: float = 10.0
    efficientnet_n_mels: int = 128
    efficientnet_hop_length: int = 512     # Training: 512
    efficientnet_n_fft: int = 512          # Training: 512
    efficientnet_max_time: int = 1024
    
    # Klassennamen
    class_map_path: str = "class_map.json"
    
    # Inferenz
    top_k: int = 5  # Top-K Vorhersagen


# ============================================================================
# INTENT-MODELL (TFLite) - "Ist da ein Vogel?"
# ============================================================================

class IntentDetector:
    """Schnelle Vogel/Nicht-Vogel Erkennung mit TFLite."""
    
    def __init__(self, config: Config):
        self.config = config
        self.interpreter = None
        self._load_model()
    
    def _load_model(self):
        """Lädt das TFLite-Modell."""
        try:
            import tflite_runtime.interpreter as tflite
            self.interpreter = tflite.Interpreter(model_path=self.config.intent_model_path)
        except ImportError:
            # Fallback auf TensorFlow Lite
            import tensorflow as tf
            self.interpreter = tf.lite.Interpreter(model_path=self.config.intent_model_path)
        
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        print(f"✅ Intent-Modell geladen: {self.config.intent_model_path}")
        print(f"   Input Shape: {self.input_details[0]['shape']}")
    
    def preprocess(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Erzeugt Mel-Spektrogramm für Intent-Modell."""
        # Resampling falls nötig
        if sr != self.config.intent_sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.config.intent_sample_rate)
            sr = self.config.intent_sample_rate
        
        # Auf 2 Sekunden normalisieren
        target_samples = int(self.config.intent_sample_rate * self.config.intent_duration)
        if len(audio) > target_samples:
            audio = audio[:target_samples]
        else:
            audio = np.pad(audio, (0, target_samples - len(audio)))
        
        # Mel-Spektrogramm
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=self.config.intent_n_mels,
            hop_length=self.config.intent_hop_length,
            fmax=8000
        )
        
        # Log-Skalierung und Normalisierung
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_norm = (mel_spec_db + 80) / 80
        mel_spec_norm = np.clip(mel_spec_norm, 0, 1)
        
        # Shape: (64, 63, 1) -> (1, 64, 63, 1)
        return mel_spec_norm[..., np.newaxis][np.newaxis, ...].astype(np.float32)
    
    def predict(self, audio: np.ndarray, sr: int) -> Tuple[bool, float]:
        """
        Erkennt ob ein Vogel im Audio ist.
        
        Returns:
            (is_bird, confidence): Tuple aus bool und Konfidenz 0-1
        """
        mel = self.preprocess(audio, sr)
        
        self.interpreter.set_tensor(self.input_details[0]['index'], mel)
        self.interpreter.invoke()
        
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        confidence = float(output[0][0])
        
        is_bird = confidence >= self.config.intent_threshold
        return is_bird, confidence


# ============================================================================
# EFFICIENTNET (ONNX) - "Welche Vogelart?"
# ============================================================================

class SpeciesClassifier:
    """EfficientNet-B0 basierte Vogelarten-Klassifikation mit ONNX Runtime."""
    
    def __init__(self, config: Config):
        self.config = config
        self.session = None
        self.class_names = None
        self._load_model()
        self._load_class_map()
    
    def _load_model(self):
        """Lädt das ONNX-Modell."""
        import onnxruntime as ort
        
        # Prüfe auf verfügbare Execution Providers
        providers = ['CPUExecutionProvider']
        available = ort.get_available_providers()
        
        if 'CUDAExecutionProvider' in available:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            print("   🚀 CUDA-Beschleunigung verfügbar!")
        
        self.session = ort.InferenceSession(
            self.config.efficientnet_model_path,
            providers=providers
        )
        
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        print(f"✅ EfficientNet geladen: {self.config.efficientnet_model_path}")
        print(f"   Input: {self.session.get_inputs()[0].shape}")
        print(f"   Provider: {self.session.get_providers()[0]}")
    
    def _load_class_map(self):
        """Lädt die Klassennamen."""
        with open(self.config.class_map_path, 'r') as f:
            class_map = json.load(f)
        
        # Invertiere: {name: id} -> {id: name}
        self.class_names = {v: k for k, v in class_map.items()}
        self.num_classes = len(self.class_names)
        print(f"   Klassen: {self.num_classes} Vogelarten")
    
    def _format_species_name(self, scientific_name: str) -> str:
        """Formatiert wissenschaftlichen Namen schöner."""
        return scientific_name.replace('_', ' ')
    
    def preprocess(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Erzeugt Mel-Spektrogramm für EfficientNet.
        
        WICHTIG: Parameter müssen exakt mit prepare_data_from_raw_audio.py übereinstimmen!
        Training: SR=32000, n_mels=128, n_fft=512, hop_length=512
        Normalisierung: dB clipped [-80, 0] -> uint8 [0, 255] -> float [0, 1]
        """
        # Resampling auf 32kHz (wie im Training)
        if sr != self.config.efficientnet_sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.config.efficientnet_sample_rate)
            sr = self.config.efficientnet_sample_rate
        
        # Mel-Spektrogramm berechnen (exakt wie im Training)
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=self.config.efficientnet_n_mels,
            n_fft=self.config.efficientnet_n_fft,
            hop_length=self.config.efficientnet_hop_length
        )
        
        # Log-Skalierung (wie im Training)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Clipping und Normalisierung (EXAKT wie im Training!)
        mel_db = np.clip(mel_db, -80, 0)
        mel_uint8 = ((mel_db + 80) / 80 * 255).astype(np.float32)
        mel_final = mel_uint8 / 255.0
        
        # Zeit-Dimension anpassen (padding/truncating auf 1024)
        if mel_final.shape[1] > self.config.efficientnet_max_time:
            mel_final = mel_final[:, :self.config.efficientnet_max_time]
        else:
            pad_width = self.config.efficientnet_max_time - mel_final.shape[1]
            mel_final = np.pad(mel_final, ((0, 0), (0, pad_width)), mode='constant')
        
        # Shape: (128, 1024) -> (1, 1, 128, 1024)
        return mel_final[np.newaxis, np.newaxis, ...].astype(np.float32)
    
    def predict(self, audio: np.ndarray, sr: int) -> List[Dict]:
        """
        Klassifiziert die Vogelart.
        
        Returns:
            Liste von Dicts mit top_k Vorhersagen:
            [{'species': str, 'confidence': float, 'rank': int}, ...]
        """
        mel = self.preprocess(audio, sr)
        
        # ONNX Inferenz
        outputs = self.session.run([self.output_name], {self.input_name: mel})
        logits = outputs[0][0]
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()
        
        # Top-K
        top_indices = np.argsort(probs)[::-1][:self.config.top_k]
        
        results = []
        for rank, idx in enumerate(top_indices):
            results.append({
                'rank': rank + 1,
                'species': self._format_species_name(self.class_names[idx]),
                'species_scientific': self.class_names[idx],
                'confidence': float(probs[idx]),
                'class_id': int(idx)
            })
        
        return results


# ============================================================================
# KOMBINIERTE PIPELINE
# ============================================================================

class Zwitscherkasten:
    """
    Zweistufige Vogelerkennungs-Pipeline.
    
    1. Intent-Check: Schnelle Prüfung ob überhaupt ein Vogel da ist
    2. Klassifikation: Detaillierte Artenbestimmung
    
    Diese Architektur spart Rechenzeit auf dem Raspberry Pi,
    da das große EfficientNet nur bei Bedarf läuft.
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        
        print("\n" + "="*60)
        print("🐦 ZWITSCHERKASTEN - Vogelstimmenerkennung")
        print("="*60 + "\n")
        
        # Modelle laden
        print("📦 Lade Modelle...")
        self.intent_detector = IntentDetector(self.config)
        self.species_classifier = SpeciesClassifier(self.config)
        
        print("\n✅ Zwitscherkasten bereit!\n")
    
    def analyze(self, audio_path: str, skip_intent: bool = False) -> Dict:
        """
        Analysiert eine Audio-Datei.
        
        Args:
            audio_path: Pfad zur Audio-Datei
            skip_intent: True um Intent-Check zu überspringen (direkt klassifizieren)
        
        Returns:
            Dict mit Analyse-Ergebnis
        """
        result = {
            'audio_file': audio_path,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'bird_detected': False,
            'intent_confidence': None,
            'species': None,
            'predictions': [],
            'processing_time_ms': {}
        }
        
        # Audio laden
        start_load = time.perf_counter()
        audio, sr = librosa.load(audio_path, sr=None)
        result['processing_time_ms']['audio_load'] = (time.perf_counter() - start_load) * 1000
        result['audio_duration_s'] = len(audio) / sr
        result['audio_sample_rate'] = sr
        
        # Stufe 1: Intent-Check
        if not skip_intent:
            start_intent = time.perf_counter()
            is_bird, intent_conf = self.intent_detector.predict(audio, sr)
            result['processing_time_ms']['intent_check'] = (time.perf_counter() - start_intent) * 1000
            result['intent_confidence'] = intent_conf
            result['bird_detected'] = is_bird
            
            if not is_bird:
                result['processing_time_ms']['total'] = sum(result['processing_time_ms'].values())
                return result
        else:
            result['bird_detected'] = True
            result['intent_confidence'] = 1.0
        
        # Stufe 2: Artenklassifikation
        start_classify = time.perf_counter()
        predictions = self.species_classifier.predict(audio, sr)
        result['processing_time_ms']['classification'] = (time.perf_counter() - start_classify) * 1000
        
        result['predictions'] = predictions
        result['species'] = predictions[0] if predictions else None
        
        result['processing_time_ms']['total'] = sum(result['processing_time_ms'].values())
        
        return result
    
    def analyze_stream(self, audio_chunk: np.ndarray, sr: int) -> Dict:
        """
        Analysiert einen Audio-Chunk (für Echtzeit-Streaming).
        
        Args:
            audio_chunk: NumPy Array mit Audio-Daten
            sr: Sample Rate
        
        Returns:
            Dict mit Analyse-Ergebnis
        """
        result = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'bird_detected': False,
            'intent_confidence': None,
            'species': None,
            'predictions': [],
            'processing_time_ms': {}
        }
        
        # Stufe 1: Intent-Check
        start_intent = time.perf_counter()
        is_bird, intent_conf = self.intent_detector.predict(audio_chunk, sr)
        result['processing_time_ms']['intent_check'] = (time.perf_counter() - start_intent) * 1000
        result['intent_confidence'] = intent_conf
        result['bird_detected'] = is_bird
        
        if not is_bird:
            result['processing_time_ms']['total'] = sum(result['processing_time_ms'].values())
            return result
        
        # Stufe 2: Artenklassifikation
        start_classify = time.perf_counter()
        predictions = self.species_classifier.predict(audio_chunk, sr)
        result['processing_time_ms']['classification'] = (time.perf_counter() - start_classify) * 1000
        
        result['predictions'] = predictions
        result['species'] = predictions[0] if predictions else None
        
        result['processing_time_ms']['total'] = sum(result['processing_time_ms'].values())
        
        return result
    
    def print_result(self, result: Dict):
        """Gibt das Ergebnis schön formatiert aus."""
        print("\n" + "-"*50)
        print(f"📁 Datei: {result.get('audio_file', 'Stream')}")
        print(f"⏱️  Dauer: {result.get('audio_duration_s', 0):.2f}s")
        print("-"*50)
        
        intent_emoji = "✅" if result['bird_detected'] else "❌"
        print(f"\n🎯 Vogel erkannt: {intent_emoji} ({result['intent_confidence']:.1%})")
        
        if result['bird_detected'] and result['predictions']:
            print(f"\n🔬 Top-{len(result['predictions'])} Vorhersagen:")
            for pred in result['predictions']:
                bar_len = int(pred['confidence'] * 20)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                print(f"   {pred['rank']}. {pred['species']}")
                print(f"      [{bar}] {pred['confidence']:.1%}")
        
        print(f"\n⚡ Laufzeiten:")
        for key, ms in result['processing_time_ms'].items():
            print(f"   {key}: {ms:.1f} ms")
        
        print("-"*50 + "\n")


# ============================================================================
# DEMO / TEST
# ============================================================================

def demo():
    """Demonstriert die Pipeline."""
    
    # Initialisierung
    zk = Zwitscherkasten()
    
    # Test mit einer Datei (falls vorhanden)
    test_files = list(Path('.').glob('*.mp3')) + list(Path('.').glob('*.wav'))
    
    if test_files:
        print(f"🎵 Gefundene Audio-Dateien: {len(test_files)}")
        for f in test_files[:3]:  # Maximal 3 testen
            result = zk.analyze(str(f))
            zk.print_result(result)
    else:
        print("ℹ️  Keine Audio-Dateien im aktuellen Verzeichnis gefunden.")
        print("    Nutze zk.analyze('pfad/zu/audio.mp3') oder")
        print("    zk.analyze_stream(audio_array, sample_rate) für Streaming.")
    
    # Zeige Beispiel-Code
    print("\n" + "="*60)
    print("📝 BEISPIEL-NUTZUNG")
    print("="*60)
    print("""
from zwitscherkasten_inference import Zwitscherkasten, Config

# Standard-Konfiguration
zk = Zwitscherkasten()

# Datei analysieren
result = zk.analyze("vogelgesang.mp3")
zk.print_result(result)

# Oder direkt die Top-Art abrufen:
if result['bird_detected']:
    print(f"Erkannt: {result['species']['species']}")
    print(f"Konfidenz: {result['species']['confidence']:.1%}")

# Für Echtzeit-Streaming (z.B. vom Mikrofon):
import sounddevice as sd

def callback(indata, frames, time, status):
    audio = indata[:, 0]  # Mono
    result = zk.analyze_stream(audio, 16000)
    if result['bird_detected']:
        print(f"🐦 {result['species']['species']}")

# Konfiguration anpassen:
config = Config()
config.intent_threshold = 0.7  # Höhere Schwelle = weniger False Positives
config.top_k = 3  # Nur Top-3 zeigen
zk = Zwitscherkasten(config)
""")


if __name__ == "__main__":
    demo()