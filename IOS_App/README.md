# ZwitscherkastenApp

ZwitscherkastenApp is an open-source, native iOS application designed for real-time bird identification via audio analysis and visual recognition. The app leverages a cascading machine learning pipeline, on-device signal processing using the Accelerate framework, and a modern SwiftUI user interface.

## Overview

This project demonstrates the implementation of a high-performance, offline-capable identification system. It processes audio streams to generate Mel-spectrograms in real-time and classifies bird species using optimized CoreML models.

### Key Features

* **Real-Time Audio Analysis:** Generates Mel-spectrograms directly on the device using vDSP (Accelerate framework) and FFT.
* **Cascading ML Pipeline:**
    1.  **Intent Detection:** A lightweight model (BirdIntent) filters noise and silence to prevent false positives.
    2.  **Classification:** A specialized model (ZwitscherNet) identifies the specific bird species upon positive intent detection.
* **Visual Recognition:** Integration of Camera and Photo Library for image-based classification using a custom CoreML pipeline with advanced preprocessing (Center Crop, Normalization) matching Python training standards.
* **Session Management:** Groups detections into recording sessions with timestamp and GPS location tagging.
* **Smart History:** Local persistence of detection data with playback functionality and Apple Maps integration.

## Technical Stack

The application is built with a focus on performance and thread safety, utilizing the latest Swift concurrency features.

* **Language:** Swift 6
* **UI Framework:** SwiftUI (MVVM architecture)
* **Audio:** AVFoundation, AVAudioRecorder (PCM Float32, 16kHz/32kHz)
* **Signal Processing:** Accelerate (vDSP) for Fast Fourier Transform (FFT), Hann windowing, and Mel-filter bank application.
* **Machine Learning:** CoreML (Direct Inference), Accelerate (Softmax implementation).
* **Image Processing:** CoreGraphics & VideoToolbox (Manual pixel buffer conversion & preprocessing).
* **Concurrency:** Structured Concurrency (`@MainActor`, `Task`, `nonisolated`, `Sendable`).

## Installation and Setup

### Prerequisites

* Xcode 15.0 or later
* iOS 17.0 or later (Physical device recommended for microphone and GPS testing)

### Getting Started

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/your-username/zwitscherkasten.git](https://github.com/your-username/zwitscherkasten.git)
    cd zwitscherkasten
    ```

2.  **Verify Resources**
    Ensure the following JSON filter banks are present in the "Copy Bundle Resources" phase:
    * `mel_filters.json` (For the primary classifier spectrograms)
    * `mel_filters_intent.json` (For the intent model spectrograms)

3.  **Build and Run**
    Select your target device and run the application via Xcode.

## Project Structure

* **Managers/**: Handles system interactions.
    * `AudioInputManager.swift`: Orchestrates the recording loop and ML pipeline.
    * `LocationManager.swift`: Wrapper for CoreLocation updates.
    * `AudioPlayer.swift`: Manages playback of recorded samples.
* **ML/**: Signal processing and inference logic.
    * `SpectrogramGenerator.swift`: vDSP implementation for spectrogram generation.
    * `ImageClassifier.swift`: Direct CoreML implementation for image analysis with custom preprocessing.
    * `AudioClassifier.swift`: (Internal) Handles audio model inference.
* **Models/**: Data structures.
    * `Bird.swift`: Static species data and logic.
    * `DetectionItem.swift`: Codable structures for persistence.
* **Views/**: SwiftUI interface components.
    * `AudioScannerView.swift`: Real-time visualization and control.
    * `VisualScannerView.swift`: Camera and library image handling with live result feedback.
