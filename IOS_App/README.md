# ZwitscherkastenApp

ZwitscherkastenApp is an open-source, native iOS application designed for real-time bird identification via audio analysis and visual recognition. The app leverages a cascading machine learning pipeline, on-device signal processing using the Accelerate framework, and a modern SwiftUI user interface.

## Overview

This project demonstrates the implementation of a high-performance, offline-capable identification system. It processes audio streams to generate Mel-spectrograms in real-time and classifies bird species using CoreML.

### Key Features

* **Real-Time Audio Analysis:** Generates Mel-spectrograms directly on the device using vDSP (Accelerate framework) and FFT.
* **Cascading ML Pipeline:**
    1.  **Intent Detection:** A lightweight model (BirdIntent) filters noise and silence to prevent false positives.
    2.  **Classification:** A specialized model (ZwitscherNet) identifies the specific bird species upon positive intent detection.
* **Visual Recognition:** Integration of Camera and Photo Library for image-based classification using the Vision framework.
* **Session Management:** Groups detections into recording sessions with timestamp and GPS location tagging.
* **Smart History:** Local persistence of detection data with playback functionality and Apple Maps integration.

## Technical Stack

The application is built with a focus on performance and thread safety, utilizing the latest Swift concurrency features.

* **Language:** Swift 6
* **UI Framework:** SwiftUI (MVVM architecture)
* **Audio:** AVFoundation, AVAudioRecorder (PCM Float32, 16kHz/32kHz)
* **Signal Processing:** Accelerate (vDSP) for Fast Fourier Transform (FFT), Hann windowing, and Mel-filter bank application.
* **Machine Learning:** CoreML, Vision (VNCoreMLRequest).
* **Concurrency:** Structured Concurrency (`@MainActor`, `Task`, `nonisolated`, `Sendable`).

## Installation and Setup

### Prerequisites

* Xcode 15.0 or later
* iOS 17.0 or later (Physical device recommended for microphone and GPS testing)

### Getting Started

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/your-username/ZwitscherkastenApp.git](https://github.com/your-username/ZwitscherkastenApp.git)
    cd ZwitscherkastenApp
    ```

2.  **Add Machine Learning Models**
    Due to file size limits, the CoreML models are not included in the source code. You must add the following files to your project target:
    * `ZwitscherNet.mlmodel` (Primary Classifier)
    * `BirdIntent.mlmodel` (Intent/Gatekeeper Model)

3.  **Verify Resources**
    Ensure the following JSON filter banks are present in the "Copy Bundle Resources" phase:
    * `mel_filters.json` (For the primary classifier spectrograms)
    * `mel_filters_intent.json` (For the intent model spectrograms)

4.  **Build and Run**
    Select your target device and run the application via Xcode.

## Project Structure

* **Managers/**: Handles system interactions.
    * `AudioInputManager.swift`: Orchestrates the recording loop and ML pipeline.
    * `LocationManager.swift`: Wrapper for CoreLocation updates.
    * `AudioPlayer.swift`: Manages playback of recorded samples.
* **ML/**: Signal processing and inference logic.
    * `SpectrogramGenerator.swift`: vDSP implementation for spectrogram generation.
    * `BirdClassifier.swift`: Interface for Vision framework requests.
* **Models/**: Data structures.
    * `Bird.swift`: Static species data and logic.
    * `DetectionItem.swift`: Codable structures for persistence.
* **Views/**: SwiftUI interface components.
    * `AudioScannerView.swift`: Real-time visualization and control.
    * `VisualScannerView.swift`: Camera and library image handling.

## Privacy Policy

ZwitscherkastenApp processes all audio and visual data locally on the device (On-Device Inference). No audio recordings, images, or location data are transmitted to external servers.

## License

This project is open source and available under the MIT License. See the LICENSE file for more details.
