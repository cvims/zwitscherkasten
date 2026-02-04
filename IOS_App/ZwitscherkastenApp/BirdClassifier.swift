import CoreML
import Vision
import UIKit

class BirdClassifier {
    
    // Initializes the ZwitscherNet CoreML model.
    let model: ZwitscherNet? = {
        do {
            let config = MLModelConfiguration()
            return try ZwitscherNet(configuration: config)
        } catch {
            return nil
        }
    }()
    
    func predictResult(image: UIImage) -> (label: String, confidence: Double)? {
        guard let model = model else { return nil }
        
        // Resizes image to expected model input dimensions: 300x128 (WxH).
        let targetSize = CGSize(width: 300, height: 128)
        
        guard let resizedImage = image.resizeTo(size: targetSize),
              let pixelBuffer = resizedImage.toCVPixelBuffer() else {
            return nil
        }
        
        do {
            // Performs prediction using the 'mel_spectrogram' input key.
            let output = try model.prediction(mel_spectrogram: pixelBuffer)
            
            // Extracts class label and confidence directly from model output.
            let bestBird = output.classLabel
            let confidence = output.classLabel_probs[bestBird] ?? 0.0
            
            return (bestBird, confidence)
            
        } catch {
            print("Prediction Error: \(error)")
            return nil
        }
    }
}

// UIImage extensions for resizing and pixel buffer conversion.
extension UIImage {
    func resizeTo(size: CGSize) -> UIImage? {
        UIGraphicsBeginImageContextWithOptions(size, false, 0.0)
        self.draw(in: CGRect(origin: .zero, size: size))
        let newImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return newImage
    }
    
    func toCVPixelBuffer() -> CVPixelBuffer? {
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue
        ] as CFDictionary
        
        var pixelBuffer : CVPixelBuffer?
        
        // Creates a 1-channel grayscale pixel buffer (OneComponent8).
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            Int(self.size.width),
            Int(self.size.height),
            kCVPixelFormatType_OneComponent8,
            attrs,
            &pixelBuffer
        )
        
        guard (status == kCVReturnSuccess), let buffer = pixelBuffer else { return nil }
        
        CVPixelBufferLockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))
        let pixelData = CVPixelBufferGetBaseAddress(buffer)
        
        // Configures grayscale color space.
        let grayColorSpace = CGColorSpaceCreateDeviceGray()
        
        let context = CGContext(
            data: pixelData,
            width: Int(self.size.width),
            height: Int(self.size.height),
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: grayColorSpace,
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        )
        
        context?.translateBy(x: 0, y: self.size.height)
        context?.scaleBy(x: 1.0, y: -1.0)
        
        UIGraphicsPushContext(context!)
        self.draw(in: CGRect(x: 0, y: 0, width: self.size.width, height: self.size.height))
        UIGraphicsPopContext()
        
        CVPixelBufferUnlockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))
        
        return buffer
    }
}
