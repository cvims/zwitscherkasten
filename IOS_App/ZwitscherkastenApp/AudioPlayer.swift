import Foundation
import AVFoundation
import SwiftUI
import Combine

// Manages audio playback state and AVFoundation integration.
class AudioPlayer: NSObject, ObservableObject, AVAudioPlayerDelegate {
    
    var audioPlayer: AVAudioPlayer?
    @Published var isPlaying = false
    // Tracks the specific file URL to enable row-specific UI toggling.
    @Published var currentlyPlayingURL: URL?
    
    func play(url: URL) {
        // Toggles playback off if the requested URL is already active.
        if isPlaying && currentlyPlayingURL == url {
            stop()
            return
        }
        
        do {
            // Configures the audio session for explicit playback.
            try AVAudioSession.sharedInstance().setCategory(.playback, mode: .default)
            try AVAudioSession.sharedInstance().setActive(true)
            
            audioPlayer = try AVAudioPlayer(contentsOf: url)
            audioPlayer?.delegate = self
            audioPlayer?.prepareToPlay()
            audioPlayer?.play()
            
            withAnimation {
                isPlaying = true
                currentlyPlayingURL = url
            }
        } catch {
            print("Fehler beim Abspielen: \(error)")
        }
    }
    
    func stop() {
        audioPlayer?.stop()
        withAnimation {
            isPlaying = false
            currentlyPlayingURL = nil
        }
    }
    
    // AVAudioPlayerDelegate: Resets state when playback finishes naturally.
    func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        stop()
    }
}
