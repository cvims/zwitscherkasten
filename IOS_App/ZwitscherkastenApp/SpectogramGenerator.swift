import Foundation
import CoreGraphics
import Accelerate
import AVFoundation

final class SpectrogramGenerator {
    
    // MARK: - Configuration
    static let targetRate = 32000.0
    static let nFFT = 512
    static let hopLength = 512
    static let targetWidth = 300
    static let targetHeight = 128
    
    // Cache
    private static var melFilterBank: [Float]?
    private static var melRows = 0
    private static var melCols = 0
    private static var windowSequence: [Float]?
    
    // Generates a thread-safe CGImage from the audio URL.
    static func generateMelSpecImage(from audioURL: URL) -> CGImage? {
        
        // 1. Load Resources
        if melFilterBank == nil {
            if !loadMelFilters() {
                print("SpectrogramGenerator: Abort, filter bank could not be loaded.")
                return nil
            }
            setupWindow()
        }
        
        // 2. Load & Resample Audio
        guard let pcmData = loadAndResampleAudio(from: audioURL, targetRate: targetRate) else {
            print("SpectrogramGenerator: Audio could not be loaded/resampled: \(audioURL.lastPathComponent)")
            return nil
        }
        
        guard let melMatrix = melFilterBank, let window = windowSequence else {
            print("SpectrogramGenerator: Internal data missing (Matrix or Window).")
            return nil
        }
        
        // 3. FFT Setup
        let log2n = vDSP_Length(log2(Float(nFFT)))
        guard let fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else { return nil }
        defer { vDSP_destroy_fftsetup(fftSetup) }
        
        var allMelColumns: [Float] = []
        let windowCount = (pcmData.count - nFFT) / hopLength + 1
        let limit = min(windowCount, targetWidth)
        
        var realPart = [Float](repeating: 0, count: nFFT/2)
        var imagPart = [Float](repeating: 0, count: nFFT/2)
        
        realPart.withUnsafeMutableBufferPointer { realPtr in
            imagPart.withUnsafeMutableBufferPointer { imagPtr in
                guard let realBase = realPtr.baseAddress, let imagBase = imagPtr.baseAddress else { return }
                var splitComplex = DSPSplitComplex(realp: realBase, imagp: imagBase)
                
                for i in 0..<limit {
                    let start = i * hopLength
                    if start + nFFT > pcmData.count { break }
                    
                    var chunk = Array(pcmData[start ..< start + nFFT])
                    vDSP_vmul(chunk, 1, window, 1, &chunk, 1, vDSP_Length(nFFT))
                    
                    chunk.withUnsafeBytes { bufferPtr in
                        let complexPtr = bufferPtr.baseAddress!.bindMemory(to: DSPComplex.self, capacity: nFFT/2)
                        vDSP_ctoz(complexPtr, 2, &splitComplex, 1, vDSP_Length(nFFT/2))
                    }
                    vDSP_fft_zrip(fftSetup, &splitComplex, 1, log2n, FFTDirection(kFFTDirection_Forward))
                    
                    var magnitudes = [Float](repeating: 0.0, count: nFFT/2)
                    vDSP_zvmags(&splitComplex, 1, &magnitudes, 1, vDSP_Length(nFFT/2))
                    magnitudes.append(0.0) // Nyquist Bin
                    
                    var melResult = [Float](repeating: 0.0, count: melRows)
                    vDSP_mmul(melMatrix, 1, magnitudes, 1, &melResult, 1, vDSP_Length(melRows), 1, vDSP_Length(melCols))
                    
                    allMelColumns.append(contentsOf: melResult)
                }
            }
        }
        
        // 4. Auto-Leveling
        var currentMaxVal: Float = 1e-9
        if !allMelColumns.isEmpty {
            vDSP_maxv(allMelColumns, 1, &currentMaxVal, vDSP_Length(allMelColumns.count))
        }
        
        // 5. Pixel Generation
        var pixelData = [UInt8](repeating: 0, count: targetWidth * targetHeight)
        
        for x in 0..<limit {
            for y in 0..<targetHeight {
                let index = (x * targetHeight) + y
                if index >= allMelColumns.count { break }
                
                let val = allMelColumns[index]
                var db = 10.0 * log10(max(val, 1e-10) / currentMaxVal)
                
                if db < -80 { db = -80 }
                if db > 0 { db = 0 }
                
                let pixelVal = UInt8((db + 80) / 80 * 255)
                
                // Rotation correction (Row=Frequency, Column=Time)
                let targetRow = y
                let targetIndex = (targetRow * targetWidth) + x
                
                if targetIndex < pixelData.count {
                    pixelData[targetIndex] = pixelVal
                }
            }
        }
        
        return createCGImage(pixels: pixelData, width: targetWidth, height: targetHeight)
    }
    
    
    // MARK: - Helper: Resize (CoreGraphics)
    static func resizeImage(_ cgImage: CGImage, to targetSize: CGSize) -> CGImage? {
        let width = Int(targetSize.width)
        let height = Int(targetSize.height)
        let bitsPerComponent = cgImage.bitsPerComponent
        let bytesPerRow = 0
        let colorSpace = cgImage.colorSpace ?? CGColorSpaceCreateDeviceGray()
        let bitmapInfo = cgImage.bitmapInfo
        
        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: bitsPerComponent,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: bitmapInfo.rawValue
        ) else { return nil }
        
        context.interpolationQuality = .high
        context.draw(cgImage, in: CGRect(origin: .zero, size: targetSize))
        
        return context.makeImage()
    }
    
    // MARK: - Audio Loading
    private static func loadAndResampleAudio(from url: URL, targetRate: Double) -> [Float]? {
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
            if tempBuffer.frameLength == 0 {
                outStatus.pointee = .endOfStream; return nil
            } else {
                outStatus.pointee = .haveData; return tempBuffer
            }
        }
        
        converter.convert(to: buffer, error: &error, withInputFrom: inputBlock)
        if error != nil { return nil }
        
        guard let channelData = buffer.floatChannelData else { return nil }
        return Array(UnsafeBufferPointer(start: channelData[0], count: Int(buffer.frameLength)))
    }
    
    // MARK: - JSON & Windowing
    private static func loadMelFilters() -> Bool {
        guard let url = Bundle.main.url(forResource: "mel_filters", withExtension: "json") else {
            print("CRITICAL: 'mel_filters.json' not found! Check 'Copy Bundle Resources'.")
            return false
        }
        do {
            let data = try Data(contentsOf: url)
            guard let json = try JSONSerialization.jsonObject(with: data, options: []) as? [String: Any],
                  let rows = json["rows"] as? Int,
                  let cols = json["cols"] as? Int,
                  // Extracts filter weights
                  let rawData = json["weights"] as? [Double]
            else {
                print("Error: JSON structure invalid.")
                return false
            }
            
            self.melFilterBank = rawData.map { Float($0) }
            self.melRows = rows
            self.melCols = cols
            print("Mel filter bank loaded: \(rows)x\(cols)")
            return true
        } catch {
            print("Error parsing JSON: \(error)")
            return false
        }
    }
    
    private static func setupWindow() {
        var win = [Float](repeating: 0, count: nFFT)
        vDSP_hann_window(&win, vDSP_Length(nFFT), 0)
        self.windowSequence = win
    }
    
    private static func createCGImage(pixels: [UInt8], width: Int, height: Int) -> CGImage? {
        let colorSpace = CGColorSpaceCreateDeviceGray()
        let data = Data(pixels) as CFData
        guard let provider = CGDataProvider(data: data) else { return nil }
        
        return CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 8,
            bytesPerRow: width,
            space: colorSpace,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        )
    }
}
