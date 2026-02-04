import Foundation
import Accelerate
import AVFoundation
import CoreGraphics
import UIKit // Only for the final step (UIImage)

// Final class utilizing Sendable mechanisms for thread safety.
final class SpectrogramGenerator_Intent: @unchecked Sendable {
    
    // MARK: - Configuration
    // 'nonisolated' ensures these constants are not bound to the MainActor.
    nonisolated static let targetRate: Double = 16000.0
    nonisolated static let nFFT = 2048
    nonisolated static let hopLength = 512
    nonisolated static let melHeight = 64
    
    // MARK: - Cache (Thread-Safe Wrapper)
    // Using 'nonisolated(unsafe)' for lazy initialization to avoid Swift 6 global static variable warnings.
    nonisolated(unsafe) private static var melFilterBank: [Float]?
    nonisolated(unsafe) private static var melRows = 0
    nonisolated(unsafe) private static var melCols = 0
    nonisolated(unsafe) private static var windowSequence: [Float]?
    
    // Helper lock for thread-safe initial loading.
    nonisolated(unsafe)    private static let setupLock = NSLock()
    
    /// Generates a grayscale UIImage from the audio file.
    nonisolated static func generateMelSpecImage(from audioURL: URL) -> UIImage? {
        
        // 1. Setup: Load filters & window (Thread-Safe)
        setupLock.lock()
        if melFilterBank == nil {
            if !loadMelFilters() {
                print("JSON missing! (mel_filters_intent.json)")
                setupLock.unlock()
                return nil
            }
            setupWindow()
        }
        setupLock.unlock()
        
        // 2. Load audio & resample to 16kHz
        guard let pcmData = loadAndResampleAudio(from: audioURL, targetRate: targetRate),
              let melMatrix = melFilterBank,
              let window = windowSequence else { return nil }
        
        // --- PROCESSING ---
        
        let log2n = vDSP_Length(log2(Float(nFFT)))
        guard let fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else { return nil }
        defer { vDSP_destroy_fftsetup(fftSetup) }
        
        var allMelColumns: [Float] = []
        
        let frameCount = (pcmData.count - nFFT) / hopLength + 1
        if frameCount <= 0 { return nil }
        
        var realPart = [Float](repeating: 0, count: nFFT/2)
        var imagPart = [Float](repeating: 0, count: nFFT/2)
        
        for i in 0..<frameCount {
            let start = i * hopLength
            var chunk = Array(pcmData[start ..< start + nFFT])
            
            // Windowing
            vDSP_vmul(chunk, 1, window, 1, &chunk, 1, vDSP_Length(nFFT))
            
            // Pointer-safe FFT implementation (Swift 6)
            realPart.withUnsafeMutableBufferPointer { realPtr in
                imagPart.withUnsafeMutableBufferPointer { imagPtr in
                    guard let realBase = realPtr.baseAddress,
                          let imagBase = imagPtr.baseAddress else { return }
                    
                    var splitComplex = DSPSplitComplex(realp: realBase, imagp: imagBase)
                    
                    chunk.withUnsafeBytes { bufferPtr in
                        let complexPtr = bufferPtr.baseAddress!.bindMemory(to: DSPComplex.self, capacity: nFFT/2)
                        vDSP_ctoz(complexPtr, 2, &splitComplex, 1, vDSP_Length(nFFT/2))
                    }
                    
                    vDSP_fft_zrip(fftSetup, &splitComplex, 1, log2n, FFTDirection(kFFTDirection_Forward))
                    
                    var magnitudes = [Float](repeating: 0.0, count: nFFT/2)
                    vDSP_zvmags(&splitComplex, 1, &magnitudes, 1, vDSP_Length(nFFT/2))
                    magnitudes.append(0.0)
                    
                    var melResult = [Float](repeating: 0.0, count: melRows)
                    vDSP_mmul(melMatrix, 1, magnitudes, 1, &melResult, 1, vDSP_Length(melRows), 1, vDSP_Length(melCols))
                    
                    allMelColumns.append(contentsOf: melResult)
                }
            }
        }
        
        // --- NORMALIZATION ---
        var currentMaxVal: Float = 1e-9
        vDSP_maxv(allMelColumns, 1, &currentMaxVal, vDSP_Length(allMelColumns.count))
        
        let width = frameCount
        let height = melRows
        var pixelData = [UInt8](repeating: 0, count: width * height)
        
        for col in 0..<width {
            for row in 0..<height {
                let rawIndex = col * height + row
                let val = allMelColumns[rawIndex]
                var db = 10.0 * log10(max(val, 1e-10) / currentMaxVal)
                if db < -80 { db = -80 }
                if db > 0 { db = 0 }
                let pixelVal = UInt8((db + 80) / 80 * 255)
                
                let targetRow = height - 1 - row
                let pixelIndex = targetRow * width + col
                if pixelIndex < pixelData.count { pixelData[pixelIndex] = pixelVal }
            }
        }
        
        return createBitmapImage(pixels: pixelData, width: width, height: height)
    }
    
    // MARK: - Helpers (All nonisolated)
    
    nonisolated private static func loadAndResampleAudio(from url: URL, targetRate: Double) -> [Float]? {
        guard let file = try? AVAudioFile(forReading: url) else { return nil }
        guard let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: targetRate, channels: 1, interleaved: false) else { return nil }
        
        let ratio = targetRate / file.processingFormat.sampleRate
        let capacity = AVAudioFrameCount(Double(file.length) * ratio) + 8192
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: capacity) else { return nil }
        guard let converter = AVAudioConverter(from: file.processingFormat, to: format) else { return nil }
        
        converter.sampleRateConverterQuality = AVAudioQuality.max.rawValue
        
        var error: NSError? = nil
        let inputBlock: AVAudioConverterInputBlock = { inNumPackets, outStatus in
            let tempBuffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: inNumPackets)!
            try? file.read(into: tempBuffer)
            if tempBuffer.frameLength == 0 { outStatus.pointee = .endOfStream; return nil }
            else { outStatus.pointee = .haveData; return tempBuffer }
        }
        converter.convert(to: buffer, error: &error, withInputFrom: inputBlock)
        guard let channelData = buffer.floatChannelData else { return nil }
        return Array(UnsafeBufferPointer(start: channelData[0], count: Int(buffer.frameLength)))
    }
    
    nonisolated private static func loadMelFilters() -> Bool {
        guard let url = Bundle.main.url(forResource: "mel_filters_intent", withExtension: "json") else { return false }
        do {
            let data = try Data(contentsOf: url)
            guard let json = try JSONSerialization.jsonObject(with: data, options: []) as? [String: Any],
                  let rows = json["rows"] as? Int,
                  let cols = json["cols"] as? Int,
                  let rawWeights = json["weights"] as? [Double] else { return false }
            self.melFilterBank = rawWeights.map { Float($0) }
            self.melRows = rows
            self.melCols = cols
            return true
        } catch { return false }
    }
    
    nonisolated private static func setupWindow() {
        var win = [Float](repeating: 0, count: nFFT)
        vDSP_hann_window(&win, vDSP_Length(nFFT), 0)
        self.windowSequence = win
    }
    
    nonisolated private static func createBitmapImage(pixels: [UInt8], width: Int, height: Int) -> UIImage? {
        let colorSpace = CGColorSpaceCreateDeviceGray()
        let data = Data(pixels) as CFData
        guard let provider = CGDataProvider(data: data) else { return nil }
        guard let cgImage = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 8, bytesPerRow: width, space: colorSpace, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue), provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent) else { return nil }
        return UIImage(cgImage: cgImage)
    }
}
