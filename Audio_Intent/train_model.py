import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
import glob

# --- EINSTELLUNGEN ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "processed_data")
MODEL_SAVE_PATH = "bird_intent_model.h5"

# Exakt die Zahlen, die dein Preprocessing ausgespuckt hat:
INPUT_SHAPE = (64, 63, 1) 

def load_data():
    """Lädt die Daten vom Preprocessing."""
    print("------------------------------------------------")
    print("Lade Daten in den Arbeitsspeicher...")
    
    X = []
    y = []
    
    # 1. Bird (Klasse 1)
    bird_files = glob.glob(os.path.join(DATA_DIR, "bird", "*.npy"))
    print(f"-> {len(bird_files)} Vogel-Samples gefunden.")
    for f in bird_files:
        data = np.load(f)
        X.append(data)
        y.append(1.0) # 1 = Vogel
        
    # 2. Nobird (Klasse 0)
    nobird_files = glob.glob(os.path.join(DATA_DIR, "nobird", "*.npy"))
    print(f"-> {len(nobird_files)} Nicht-Vogel-Samples gefunden.")
    for f in nobird_files:
        data = np.load(f)
        X.append(data)
        y.append(0.0) # 0 = Kein Vogel

    X = np.array(X)
    y = np.array(y)
    
    # Mischen (Shuffling) ist wichtig, damit nicht erst alle Vögel und dann alle Nicht-Vögel kommen
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    print(f"Gesamt geladen: {len(X)} Samples.")
    print("------------------------------------------------")
    return X, y

def create_tiny_cnn(input_shape):
    """Unser optimiertes Tiny-Modell."""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # 1. Conv Block (Grobe Merkmale)
        layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # 2. Conv Block (Mittlere Merkmale)
        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2), # Gegen Auswendiglernen
        
        # 3. Conv Block (Feine Merkmale)
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        # Klassifizierung
        layers.GlobalAveragePooling2D(), # Macht das Modell winzig!
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid') # Wahrscheinlichkeit 0-1
    ])
    
    # Adam Optimizer mit Standard Lernrate
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Recall(name='recall')])
    return model

def plot_history(history):
    """Zeigt Kurven an."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Genauigkeit')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Fehler (Loss)')
    
    plt.tight_layout()
    plt.show()

def main():
    # 1. Laden
    X, y = load_data()
    if len(X) == 0: return

    # 2. Split (80% Training, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Bauen
    model = create_tiny_cnn(INPUT_SHAPE)
    model.summary() # Zeigt dir gleich an, wie klein das Modell ist
    
    # 4. Trainieren
    print("Starte Training (Early Stopping aktiviert)...")
    
    # Stoppt, wenn es 5 Runden lang nicht besser wird -> Spart Zeit
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Speichert immer das beste Modell während des Trainings
    checkpoint = callbacks.ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, mode='max')

    history = model.fit(
        X_train, y_train,
        epochs=30,              # Max 30 Runden
        batch_size=32,          # Passt locker in deine GPU
        validation_data=(X_test, y_test),
        callbacks=[early_stop, checkpoint]
    )
    
    print(f"FERTIG! Das beste Modell wurde gespeichert als: {MODEL_SAVE_PATH}")
    
    # 5. Anzeigen
    plot_history(history)

if __name__ == "__main__":
    main()