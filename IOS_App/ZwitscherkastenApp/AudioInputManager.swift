import Foundation
import AVFoundation
import Combine
import UIKit
import SwiftUI
import CoreML
import Vision
import CoreLocation

@MainActor
class AudioInputManager: NSObject, ObservableObject, AVAudioRecorderDelegate {
    
    // MARK: - Core Components
    var audioRecorder: AVAudioRecorder?
    var timer: Timer?
    
    // CoreML models for bird intent detection and specific classification.
    private var visionModel: VNCoreMLModel?       // For Audio Spectrograms
    private var intentModel: VNCoreMLModel?       // For Audio Detection (Bird yes/no)
    
    // NEW: The image classifier is initialized here once
    private let imageClassifier = ImageClassifier()
    
    // MARK: - Published State
    @Published var isRecording = false
    @Published var currentSessionBirds: [Bird] = []
    @Published var latestBird: Bird? = nil
    @Published var latestSpectrogram: UIImage? = nil
    
    // History observer triggers local persistence on change.
    @Published var history: [DetectionItem] = [] { didSet { saveHistory() } }
    private var currentSessionID: UUID = UUID()
    
    let locationManager = LocationManager()
    
    // Computed property aggregating statistics for the current recording session.
    var sessionStats: [BirdStatistic] {
        let total = Double(currentSessionBirds.count); if total == 0 { return [] }
        let grouped = Dictionary(grouping: currentSessionBirds, by: { $0.scientificName })
        return grouped.map { (_, birds) -> BirdStatistic in
            let latest = birds.last!; return BirdStatistic(bird: latest, count: birds.count, percentage: (Double(birds.count)/total)*100.0, latestAudioURL: latest.audioURL)
        }.sorted { $0.percentage > $1.percentage }
    }
    
    override init() { super.init(); setupAudioSession(); loadHistory() }
    
    // MARK: - Setup & Configuration
    // Initializes CoreML models (BirdIntent and ZwitscherNet) if not in preview mode.
    private func setupModels() {
        if ProcessInfo.processInfo.environment["XCODE_RUNNING_FOR_PREVIEWS"] == "1" { return }
        if visionModel != nil && intentModel != nil { return }
        
        do {
            let config = MLModelConfiguration()
            print("Lade BirdIntent...")
            let intentWrapper = try BirdIntent(configuration: config)
            self.intentModel = try VNCoreMLModel(for: intentWrapper.model)
            
            print("Lade ZwitscherNet...")
            let birdWrapper = try ZwitscherNet(configuration: config)
            self.visionModel = try VNCoreMLModel(for: birdWrapper.model)
            print("Audio-Modelle bereit.")
        } catch { print("Modell-Fehler (Audio): \(error)") }
    }
    
    func setupAudioSession() {
        try? AVAudioSession.sharedInstance().setCategory(.playAndRecord, mode: .measurement, options: .duckOthers)
        try? AVAudioSession.sharedInstance().setActive(true)
    }
    
    // MARK: - Recording Logic
    // Toggles the continuous scanning loop.
    func toggleScanning() {
        setupModels()
        if isRecording { stopScanning() } else { startScanning() }
    }
    
    // Begins recording segments; uses a Timer on the main run loop to segment audio files.
    private func startScanning() {
        isRecording = true; currentSessionID = UUID(); currentSessionBirds = []; latestBird = nil; startNextSegment()
        
        // Timer triggers segment termination and restart every 4.8 seconds.
        timer = Timer.scheduledTimer(withTimeInterval: 4.8, repeats: true) { [weak self] _ in
            DispatchQueue.main.async {
                self?.finishSegmentAndRestart()
            }
        }
    }
    
    private func stopScanning() {
        isRecording = false; timer?.invalidate(); audioRecorder?.stop(); print("Stopp.")
    }
    
    // Configures audio settings (PCM 16-bit, 32kHz) and starts the recorder.
    private func startNextSegment() {
        guard isRecording else { return }
        let settings: [String: Any] = [AVFormatIDKey: Int(kAudioFormatLinearPCM), AVSampleRateKey: 32000.0, AVNumberOfChannelsKey: 1, AVLinearPCMBitDepthKey: 16, AVLinearPCMIsBigEndianKey: false, AVLinearPCMIsFloatKey: false]
        try? audioRecorder = AVAudioRecorder(url: getFileURL(), settings: settings)
        audioRecorder?.delegate = self
        audioRecorder?.record()
    }
    
    private func finishSegmentAndRestart() {
        guard let recorder = audioRecorder, recorder.isRecording else { return }
        recorder.stop() // Triggers delegate method
    }
    
    // MARK: - AVAudioRecorderDelegate
    // Handles finished recording files by offloading analysis to a background task.
    nonisolated func audioRecorderDidFinishRecording(_ recorder: AVAudioRecorder, successfully flag: Bool) {
        Task { @MainActor [weak self] in
            guard let self = self else { return }
            if flag {
                let url = recorder.url
                self.analyzePipeline(url: url)
                if self.isRecording { self.startNextSegment() }
            }
        }
    }
    
    nonisolated private func getFileURL() -> URL { return FileManager.default.temporaryDirectory.appendingPathComponent("temp_\(UUID().uuidString).wav") }
    
    // MARK: - Analysis Pipeline
    // Orchestrates model execution and handles results. Cleans up temp files if no bird is detected.
    private func analyzePipeline(url: URL) {
        guard let mainModel = self.visionModel, let gatekeeper = self.intentModel else {
            try? FileManager.default.removeItem(at: url); return
        }
        
        Task {
            // Detached task for heavy lifting (Spectrogram generation + Inference)
            let resultData = await Task.detached(priority: .userInitiated) {
                return AudioInputManager.performCascadingAnalysis(url: url, intentModel: gatekeeper, classifierModel: mainModel)
            }.value
            
            if let data = resultData {
                let uiImage = UIImage(cgImage: data.image, scale: 1.0, orientation: .upMirrored)
                //self.latestSpectrogram = uiImage
                
                if let (label, confidence) = data.result {
                    // Threshold check for positive identification
                    if confidence > 0.2 {
                        self.processAudioDetection(scientificName: label, newAudioURL: url)
                    } else {
                        try? FileManager.default.removeItem(at: url)
                    }
                } else {
                    try? FileManager.default.removeItem(at: url)
                }
            } else {
                try? FileManager.default.removeItem(at: url)
            }
        }
    }
    
    // MARK: - Inference Worker
    // Thread-safe static method executing the two-stage model pipeline (Intent -> Classification).
    nonisolated private static func performCascadingAnalysis(url: URL, intentModel: VNCoreMLModel?, classifierModel: VNCoreMLModel) -> (image: CGImage, result: (identifier: String, confidence: Double)?)? {
        
        // ---------------------------------------------------------
        // 1. INTENT CHECK (Binary Classification: Bird vs Noise)
        // ---------------------------------------------------------
        guard let intentUIImage = SpectrogramGenerator_Intent.generateMelSpecImage(from: url),
              let intentCGImage = intentUIImage.cgImage else {
            return nil
        }
        
        var isBird = true
        if let binaryModel = intentModel {
            isBird = false
            let intentRequest = VNCoreMLRequest(model: binaryModel) { request, _ in
                if let results = request.results as? [VNCoreMLFeatureValueObservation],
                   let multiArray = results.first?.featureValue.multiArrayValue {
                    
                    let score = multiArray[0].doubleValue
                    
                    // --- LOGGING INTENT ---
                    print("\n--- INTENT ANALYSE ---")
                    print("   Score: \(String(format: "%.5f", score))")
                    print("   Schwelle: > 0.8")
                    
                    if score > 0.8 {
                        isBird = true
                        print("   Ergebnis: VOGEL DETEKTIERT")
                    } else {
                        print("   Ergebnis: KEIN VOGEL (Rauschen/Stille)")
                    }
                    print("-------------------------\n")
                }
            }
            intentRequest.imageCropAndScaleOption = .scaleFill
            let handler = VNImageRequestHandler(cgImage: intentCGImage, options: [:])
            try? handler.perform([intentRequest])
            
            if !isBird { return (intentCGImage, nil) }
        }
        
        // ---------------------------------------------------------
        // 2. CLASSIFICATION (Specific Species Identification)
        // ---------------------------------------------------------
        guard let classifierImage = SpectrogramGenerator.generateMelSpecImage(from: url) else { return nil }
        
        var analysisResult: (String, Double)? = nil
        
        let classifyRequest = VNCoreMLRequest(model: classifierModel) { request, _ in
            if let results = request.results as? [VNClassificationObservation] {
                
                // --- LOGGING KLASSIFIZIERUNG ---
                print("\n--- KLASSEN VORHERSAGE (Top 3) ---")
                let top3 = results.prefix(3)
                for (index, res) in top3.enumerated() {
                    let percentage = String(format: "%.2f", res.confidence * 100)
                    print("   \(index + 1). \(res.identifier): \(percentage)%")
                }
                print("------------------------------------\n")
                
                if let best = results.first {
                    analysisResult = (best.identifier, Double(best.confidence))
                }
            }
        }
        classifyRequest.imageCropAndScaleOption = .scaleFill
        let handler = VNImageRequestHandler(cgImage: classifierImage, options: [:])
        try? handler.perform([classifyRequest])
        
        return (classifierImage, analysisResult)
    }
    
    // MARK: - Post-Processing
    // Moves temp audio to permanent storage, attaches GPS data, and updates history.
    func processAudioDetection(scientificName: String, newAudioURL: URL) {
        guard let birdProto = allBirds.first(where: { $0.scientificName == scientificName }) else {
            try? FileManager.default.removeItem(at: newAudioURL)
            return
        }
        
        // FIX: Secure access to Document Directory
        guard let docDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else { return }
        let destURL = docDir.appendingPathComponent("bird_\(UUID().uuidString).wav")
        
        do {
            try FileManager.default.moveItem(at: newAudioURL, to: destURL)
            var newBird = birdProto
            newBird.audioURL = destURL
            
            // --- GPS CHECK ---
            var locationData: (Double, Double)? = nil
            
            if let loc = locationManager.lastLocation {
                locationData = (loc.latitude, loc.longitude)
                print("GPS gefunden: \(loc.latitude), \(loc.longitude)")
            } else {
                print("GPS noch nicht bereit (lastLocation ist nil)")
            }
            // -----------------
            
            let newItem = DetectionItem(
                sessionID: self.currentSessionID,
                bird: newBird,
                date: Date(),
                type: .audio,
                audioURL: destURL,
                location: locationData // Inject location
            )
            
            Task { @MainActor in
                withAnimation {
                    self.history.insert(newItem, at: 0)
                    self.currentSessionBirds.append(newBird)
                    self.latestBird = newBird
                }
            }
        } catch {
            print("Fehler Audio-Save: \(error)")
            try? FileManager.default.removeItem(at: newAudioURL)
        }
    }
    
    // MARK: - Visual Detection (Stub)
    // Handles manual visual entries (e.g. from camera).
    func processVisualDetection(image: UIImage) {
        print("Starte Bildanalyse...")
        
        imageClassifier.classify(image: image) { [weak self] (scientificName, confidence) in
            guard let self = self else { return }
            
            var finalBird: Bird
            
            // Do we have a name AND is it in our database?
            if let name = scientificName,
               let foundBird = allBirds.first(where: { $0.scientificName == name }) {
                print("Bild erkannt: \(foundBird.germanName) (\(Int(confidence * 100))%)")
                finalBird = foundBird
            } else {
                print("Bild nicht eindeutig erkannt.")
                finalBird = Bird(
                    scientificName: "unknown_visual",
                    germanName: "Unbekannt",
                    category: "Unbestimmt",
                    occurrence: "Lokal",
                    habitat: "Unbekannt"
                )
            }
            
            // Fetch Location
            var locationData: (Double, Double)? = nil
            if let loc = self.locationManager.lastLocation {
                locationData = (loc.latitude, loc.longitude)
            }
            
            // Create Entry
            let newItem = DetectionItem(
                sessionID: self.currentSessionID,
                bird: finalBird,
                date: Date(),
                type: .visual,
                audioURL: nil,
                image: image,
                location: locationData
            )
            
            // Update UI
            Task { @MainActor in
                withAnimation {
                    self.history.insert(newItem, at: 0)
                }
            }
        }
    }
    
    // MARK: - Data Management (CRUD)
    
    func deleteSpecificItem(_ item: DetectionItem) {
        if let url = item.audioURL { try? FileManager.default.removeItem(at: url) }
        withAnimation {
            if let index = history.firstIndex(where: { $0.id == item.id }) { history.remove(at: index) }
            if let sessionIndex = currentSessionBirds.firstIndex(where: { $0.id == item.bird.id }) { currentSessionBirds.remove(at: sessionIndex) }
        }
    }
    
    func deleteDetection(at offsets: IndexSet) {
        history.remove(atOffsets: offsets)
    }
    
    func deleteAllDetections() {
        history.removeAll()
    }
    
    // MARK: - Persistence
    // Serializes history to JSON in the document directory.
    private func saveHistory() {
        if let data = try? JSONEncoder().encode(history),
           let url = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first?.appendingPathComponent("history.json") {
            try? data.write(to: url)
        }
    }
    
    // Loads and decodes history on init.
    private func loadHistory() {
        if let url = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first?.appendingPathComponent("history.json"),
           let data = try? Data(contentsOf: url),
           let items = try? JSONDecoder().decode([DetectionItem].self, from: data) {
            self.history = items
        }
    }
}
