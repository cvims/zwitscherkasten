import os
import numpy as np
import librosa
from tqdm import tqdm

# --- EINSTELLUNGEN ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "raw_data")
OUT_DIR = os.path.join(BASE_DIR, "processed_data")

# Audio Parameter
SAMPLE_RATE = 16000     # 16kHz (Standard für Edge AI)
DURATION = 2.0          # Sekunden pro Schnipsel
N_MELS = 64             # Höhe des Bildes (Frequenz-Auflösung)
HOP_LENGTH = 512        # Breite wird ca. 63 Pixel sein
MAX_CHUNKS_PER_FILE = 5 # Maximal 5 Schnipsel pro Datei (damit lange Aufnahmen nicht dominieren)

SAMPLES_PER_CHUNK = int(SAMPLE_RATE * DURATION)

def process_and_save(file_path, save_dir, category_name):
    try:
        # Audio laden (librosa konvertiert mp3/wav automatisch)
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        
        # Audio in 2-Sekunden-Stücke schneiden
        total_samples = len(y)
        num_chunks = int(np.ceil(total_samples / SAMPLES_PER_CHUNK))
        
        # Limitieren, damit wir nicht 100 Schnipsel von einer Datei haben
        num_chunks = min(num_chunks, MAX_CHUNKS_PER_FILE)
        
        saved_chunks = 0
        
        for i in range(num_chunks):
            start = i * SAMPLES_PER_CHUNK
            end = start + SAMPLES_PER_CHUNK
            
            chunk = y[start:end]
            
            # Padding (wenn Stück kürzer als 2s)
            if len(chunk) < SAMPLES_PER_CHUNK:
                chunk = np.pad(chunk, (0, SAMPLES_PER_CHUNK - len(chunk)))
            
            # Mel Spektrogramm berechnen
            mel_spec = librosa.feature.melspectrogram(
                y=chunk, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH, fmax=8000
            )
            
            # Log-Skalierung (dB)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalisierung (-80dB bis 0dB auf 0 bis 1 skalieren)
            mel_spec_norm = (mel_spec_db + 80) / 80
            mel_spec_norm = np.clip(mel_spec_norm, 0, 1)
            
            # Dimension hinzufügen (H, W, 1) -> Wie ein Graustufenbild
            final_data = mel_spec_norm[..., np.newaxis]
            
            # Speichern (.npy ist viel schneller als Bilder beim Laden)
            filename = os.path.basename(file_path).replace('.mp3', '').replace('.wav', '')
            save_name = f"{category_name}_{filename}_chunk{i}.npy"
            np.save(os.path.join(save_dir, save_name), final_data)
            saved_chunks += 1
            
        return saved_chunks, final_data.shape

    except Exception as e:
        # Kaputte Dateien ignorieren wir einfach
        return 0, None

def main():
    # Ausgabeordner erstellen
    for cls in ["bird", "nobird"]:
        os.makedirs(os.path.join(OUT_DIR, cls), exist_ok=True)
    
    print(f"Lese Daten aus: {RAW_DIR}")
    print(f"Speichere in:  {OUT_DIR}")
    print("-" * 30)

    total_files_created = 0
    last_shape = None

    categories = ["bird", "nobird"]
    
    for category in categories:
        src_path = os.path.join(RAW_DIR, category)
        dst_path = os.path.join(OUT_DIR, category)
        
        # Alle Dateien finden
        files = [f for f in os.listdir(src_path) if f.lower().endswith(('.mp3', '.wav'))]
        print(f"Verarbeite Kategorie '{category}' ({len(files)} Quelldateien)...")
        
        for f in tqdm(files):
            file_path = os.path.join(src_path, f)
            count, shape = process_and_save(file_path, dst_path, category)
            total_files_created += count
            if shape is not None:
                last_shape = shape

    print("-" * 30)
    print("FERTIG!")
    print(f"Gesamtanzahl Trainings-Schnipsel: {total_files_created}")
    if last_shape:
        print(f"WICHTIG -> Deine Input Shape ist: {last_shape}")
        print("Merk dir diese Zahlen für das Training!")

if __name__ == "__main__":
    main()