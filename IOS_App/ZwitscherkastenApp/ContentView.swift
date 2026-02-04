import SwiftUI
import AVFoundation
import Vision

// MARK: - Main Application Entry View
struct ContentView: View {
    /// State management for audio processing and playback lifecycle
    @StateObject private var viewModel = AudioInputManager()
    @StateObject private var audioPlayer = AudioPlayer()
    
    var body: some View {
        NavigationView {
            ZStack {
                Color.backgroundSubtle.edgesIgnoringSafeArea(.all)
                
                VStack(spacing: 0) {
                    // MARK: - Header Section
                    HStack {
                        VStack(alignment: .leading) {
                            Text("Entdecken")
                                .font(.heroTitle)
                                .foregroundColor(.brandPrimary)
                            Text("Wähle deine Methode")
                                .font(.system(.subheadline, design: .rounded))
                                .foregroundColor(.textGray)
                        }
                        Spacer()
                    }
                    .padding(.horizontal, Design.Spacing.standardPadding)
                    .padding(.top, Design.Spacing.large)
                    .padding(.bottom, Design.Spacing.xLarge)
                    
                    // MARK: - Module Navigation
                    /// Entry points for Audio and Visual analysis modules
                    HStack(spacing: Design.Spacing.medium) {
                        NavigationLink(destination: AudioScannerView(viewModel: viewModel, audioPlayer: audioPlayer)) {
                            OptionCard(title: "Hören", icon: "mic.fill", color: .brandPrimary, subtitle: "Audioanalyse")
                        }
                        NavigationLink(destination: VisualScannerView(viewModel: viewModel)) {
                            OptionCard(title: "Sehen", icon: "camera.fill", color: .brandAccent, subtitle: "Fotoanalyse")
                        }
                    }
                    .padding(.horizontal, Design.Spacing.standardPadding)
                    .frame(height: Design.Size.optionCardHeight)
                    
                    // MARK: - Persistent History Header
                    HStack {
                        Text("Deine Entdeckungen")
                            .font(.sectionHeader)
                            .foregroundColor(.textDark)
                        Spacer()
                        NavigationLink(destination: FullHistoryView(viewModel: viewModel, audioPlayer: audioPlayer)) {
                            Text("Alle anzeigen")
                                .font(.subheadline)
                                .foregroundColor(.brandPrimary)
                        }
                    }
                    .padding(.horizontal, Design.Spacing.standardPadding)
                    .padding(.top, Design.Spacing.xxLarge)
                    .padding(.bottom, Design.Spacing.small)
                    
                    // MARK: - Dashboard Feed
                    /// Conditional rendering for empty states vs. recent activity (Limited to 5 items)
                    if viewModel.history.isEmpty {
                        Spacer()
                        VStack(spacing: 8) {
                            Image(systemName: "leaf")
                                .font(.largeTitle)
                                .foregroundColor(.textGray.opacity(0.3))
                            Text("Noch keine Vögel entdeckt")
                                .font(.caption)
                                .foregroundColor(.textGray)
                        }
                        Spacer()
                    } else {
                        ScrollView {
                            LazyVStack(spacing: 8) {
                                ForEach(viewModel.history.prefix(5)) { item in
                                    HistoryRow(item: item, audioPlayer: audioPlayer)
                                }
                            }
                            .padding(.horizontal, Design.Spacing.standardPadding)
                        }
                    }
                }
            }
            .navigationBarHidden(true)
        }
    }
}

// MARK: - Image Recognition View
struct VisualScannerView: View {
    @ObservedObject var viewModel: AudioInputManager
    
    /// Local UI state for image picking and analysis status
    @State private var selectedImage: UIImage?
    @State private var activeSheet: ImageSource?
    @State private var isAnalyzing = false
    
    var body: some View {
        ZStack {
            Color.white.edgesIgnoringSafeArea(.all)
            
            VStack(spacing: 20) {
                Text("Vogel analysieren")
                    .font(.cardTitle)
                    .foregroundColor(.black)
                    .padding(.top)
                
                Spacer()
                
                if let image = selectedImage {
                    // MARK: - Analysis Result View
                    VStack(spacing: 20) {
                        Image(uiImage: image)
                            .resizable()
                            .scaledToFit()
                            .frame(height: 300)
                            .cornerRadius(Design.Radius.large)
                            .shadow(radius: 5)
                        
                        /// Asynchronous state handling for Vision processing
                        if isAnalyzing {
                            VStack {
                                ProgressView()
                                    .scaleEffect(1.5)
                                    .padding()
                                Text("Analysiere...")
                                    .foregroundColor(.gray)
                            }
                        } else if let result = viewModel.history.first, result.type == .visual {
                            VStack(spacing: 10) {
                                Text("Erkannt:")
                                    .font(.caption)
                                    .foregroundColor(.gray)
                                    .textCase(.uppercase)
                                
                                Text(result.bird.germanName)
                                    .font(.largeTitle)
                                    .bold()
                                    .foregroundColor(.brandPrimary)
                                
                                Text(result.bird.scientificName)
                                    .font(.body)
                                    .italic()
                                    .foregroundColor(.gray)
                                
                                NavigationLink(destination: BirdDetailView(bird: result.bird, customImage: image)) {
                                    Text("Infos anzeigen")
                                        .padding(.horizontal, 24)
                                        .padding(.vertical, 12)
                                        .background(Color.brandAccent)
                                        .foregroundColor(.white)
                                        .cornerRadius(20)
                                }
                            }
                            .transition(.opacity)
                        }
                        
                        Spacer().frame(height: 20)
                        
                        Button("Neues Bild wählen") {
                            selectedImage = nil
                            isAnalyzing = false
                        }
                        .foregroundColor(.textGray)
                    }
                    
                } else {
                    // MARK: - Image Source Selection
                    VStack(spacing: 30) {
                        Button(action: { activeSheet = .camera }) {
                            VStack {
                                Image(systemName: "camera.fill")
                                    .font(.system(size: 50))
                                Text("Foto machen")
                                    .font(.headline)
                            }
                            .foregroundColor(.white)
                            .frame(width: 160, height: 160)
                            .background(Color.brandAccent)
                            .clipShape(Circle())
                            .shadow(color: Color.brandAccent.opacity(0.4), radius: 8, x: 0, y: 5)
                        }
                        
                        Button(action: { activeSheet = .library }) {
                            HStack {
                                Image(systemName: "photo.on.rectangle.fill")
                                Text("Aus Mediathek laden")
                                    .fontWeight(.semibold)
                            }
                            .foregroundColor(.brandAccent)
                            .padding()
                            .background(Color.brandAccent.opacity(0.1))
                            .cornerRadius(Design.Radius.large)
                        }
                    }
                }
                Spacer()
            }
        }
        /// Abstraction for UIImagePickerController
        .sheet(item: $activeSheet) { item in
            switch item {
            case .camera:
                ImagePicker(selectedImage: $selectedImage, sourceType: .camera)
                    .ignoresSafeArea()
            case .library:
                ImagePicker(selectedImage: $selectedImage, sourceType: .photoLibrary)
                    .ignoresSafeArea()
            }
        }
        /// Trigger analysis logic on image selection
        .onChange(of: selectedImage) { _, newImage in
            if let img = newImage {
                isAnalyzing = true
                viewModel.processVisualDetection(image: img)
                
                // Safety timeout for network/processing latency
                DispatchQueue.main.asyncAfter(deadline: .now() + 5.0) {
                    isAnalyzing = false
                }
            }
        }
        /// Reset loading state on data model updates
        .onChange(of: viewModel.history) { _, _ in
            withAnimation {
                isAnalyzing = false
            }
        }
    }
}

// MARK: - Domain Model Extensions
extension DetectionItem: Equatable {
    /// ID-based equality check to optimize List diffing and avoid expensive Image comparisons
    static func == (lhs: DetectionItem, rhs: DetectionItem) -> Bool {
        return lhs.id == rhs.id
    }
}

/// Identifiable abstraction for modal presentations
enum ImageSource: Identifiable {
    case camera
    case library
    var id: Int { hashValue }
}

// MARK: - History Management Views
struct FullHistoryView: View {
    @ObservedObject var viewModel: AudioInputManager
    @ObservedObject var audioPlayer: AudioPlayer
    @State private var historyMode: HistoryMode = .sessions
    @State private var showDeleteConfirmation = false
    
    enum HistoryMode: String, CaseIterable {
        case sessions = "Aufnahmen"
        case birds = "Vogelarten"
    }
    
    init(viewModel: AudioInputManager, audioPlayer: AudioPlayer) {
        self.viewModel = viewModel
        self.audioPlayer = audioPlayer
        // UIKit appearance overrides for Segmented Control
        UISegmentedControl.appearance().selectedSegmentTintColor = .white
        UISegmentedControl.appearance().backgroundColor = UIColor.systemGray5
        UISegmentedControl.appearance().setTitleTextAttributes([.foregroundColor: UIColor.black], for: .selected)
        UISegmentedControl.appearance().setTitleTextAttributes([.foregroundColor: UIColor.darkGray], for: .normal)
    }
    
    var body: some View {
        VStack(spacing: 0) {
            Picker("Anzeige", selection: $historyMode) {
                ForEach(HistoryMode.allCases, id: \.self) { mode in
                    Text(mode.rawValue).tag(mode)
                }
            }
            .pickerStyle(SegmentedPickerStyle())
            .padding()
            .background(Color.white)
            
            ZStack {
                Color(uiColor: .systemGroupedBackground).edgesIgnoringSafeArea(.all)
                if viewModel.history.isEmpty {
                    VStack(spacing: 12) {
                        Image(systemName: "tray")
                            .font(.system(size: 50))
                            .foregroundColor(.gray.opacity(0.5))
                        Text("Verlauf ist leer").font(.caption).foregroundColor(.gray)
                    }
                } else {
                    switch historyMode {
                    case .sessions: SessionListView(viewModel: viewModel, audioPlayer: audioPlayer)
                    case .birds: BirdGroupListView(viewModel: viewModel, audioPlayer: audioPlayer)
                    }
                }
            }
        }
        .navigationTitle("Deine Entdeckungen")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .navigationBarTrailing) {
                Button(role: .destructive) { showDeleteConfirmation = true } label: {
                    Image(systemName: "trash").tint(.red)
                }
                .disabled(viewModel.history.isEmpty)
            }
        }
        .alert("Alles löschen?", isPresented: $showDeleteConfirmation) {
            Button("Löschen", role: .destructive) { withAnimation { viewModel.deleteAllDetections() } }
            Button("Abbrechen", role: .cancel) { }
        } message: { Text("Möchtest du wirklich den gesamten Verlauf löschen?") }
    }
}

// MARK: - Session Grouping Logic
struct SessionListView: View {
    @ObservedObject var viewModel: AudioInputManager
    @ObservedObject var audioPlayer: AudioPlayer
    
    /// Computes dictionary grouping by SessionID and sorts by chronological order
    var groupedSessions: [(key: UUID, value: [DetectionItem])] {
        let grouped = Dictionary(grouping: viewModel.history, by: { $0.sessionID })
        return grouped.sorted { ($0.value.first?.date ?? Date.distantPast) > ($1.value.first?.date ?? Date.distantPast) }
    }
    
    var body: some View {
        List {
            ForEach(groupedSessions, id: \.key) { (sessionID, items) in
                NavigationLink(destination: SessionDetailView(items: items, viewModel: viewModel, audioPlayer: audioPlayer, title: formatDate(items.first?.date))) {
                    HStack {
                        VStack(alignment: .leading, spacing: 4) {
                            Text(formatDate(items.first?.date)).font(.headline).foregroundColor(.textDark)
                            Text("\(items.count) Vögel entdeckt").font(.caption).foregroundColor(.textGray)
                        }
                        Spacer()
                        /// Avatars for unique bird species in session
                        HStack(spacing: -8) {
                            ForEach(items.prefix(4)) { item in
                                BirdImageCircle(bird: item.bird, size: 28)
                                    .overlay(Circle().stroke(Color.backgroundSubtle, lineWidth: 1.5))
                            }
                        }
                    }
                    .padding(.vertical, 4)
                }
                .listRowBackground(Color.white)
            }
        }
        .listStyle(.insetGrouped)
        .scrollContentBackground(.hidden)
        .background(Color.backgroundSubtle)
    }
    
    func formatDate(_ date: Date?) -> String {
        guard let date = date else { return "Unbekannt" }
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "de_DE")
        formatter.dateStyle = .medium
        formatter.timeStyle = .short
        return formatter.string(from: date)
    }
}

struct BirdGroupListView: View {
    @ObservedObject var viewModel: AudioInputManager
    @ObservedObject var audioPlayer: AudioPlayer
    
    /// Groups detections by species to show frequency/collection stats
    var groupedBirds: [(key: String, value: [DetectionItem])] {
        let grouped = Dictionary(grouping: viewModel.history, by: { $0.bird.scientificName })
        return grouped.sorted { $0.value.count > $1.value.count }
    }
    var body: some View {
        List {
            ForEach(groupedBirds, id: \.key) { (sciName, items) in
                if let firstItem = items.first {
                    NavigationLink(destination: SessionDetailView(items: items, viewModel: viewModel, audioPlayer: audioPlayer, title: firstItem.bird.germanName)) {
                        HStack(spacing: 12) {
                            BirdImageCircle(bird: firstItem.bird, size: 44)
                            VStack(alignment: .leading, spacing: 4) {
                                Text(firstItem.bird.germanName).font(.headline).foregroundColor(.textDark)
                                Text("\(items.count) Aufnahmen").font(.caption).foregroundColor(.textGray)
                            }
                            Spacer()
                        }
                        .padding(.vertical, 6)
                    }
                    .listRowBackground(Color.white)
                }
            }
        }
        .listStyle(.insetGrouped)
        .scrollContentBackground(.hidden)
        .background(Color.backgroundSubtle)
    }
}

// MARK: - Specialized List Detail Views
struct SessionDetailView: View {
    let items: [DetectionItem]
    @ObservedObject var viewModel: AudioInputManager
    @ObservedObject var audioPlayer: AudioPlayer
    let title: String
    
    var body: some View {
        List {
            ForEach(items) { item in
                HistoryRow(item: item, audioPlayer: audioPlayer)
                    .listRowBackground(Color.white)
                    .listRowSeparator(.visible)
            }
            .onDelete { indexSet in
                indexSet.forEach { index in
                    guard index < items.count else { return }
                    viewModel.deleteSpecificItem(items[index])
                }
            }
        }
        .listStyle(.plain)
        .navigationTitle(title)
        .navigationBarTitleDisplayMode(.inline)
        .background(Color.backgroundSubtle)
        .toolbar {
            ToolbarItem(placement: .navigationBarTrailing) {
                EditButton().foregroundColor(.brandPrimary).fontWeight(.semibold)
            }
        }
    }
}

// MARK: - Sub-Components for Data Display
struct HistoryRow: View {
    let item: DetectionItem
    @ObservedObject var audioPlayer: AudioPlayer
    @State private var showDetail = false
    
    var body: some View {
        HStack(spacing: 12) {
            BirdImageCircle(bird: item.bird, userImage: item.image, size: 42)
            VStack(alignment: .leading, spacing: 2) {
                Text(item.bird.germanName).font(.system(size: 16, weight: .semibold)).foregroundColor(.black)
                Text(formatDate(item.date)).font(.caption).foregroundColor(.gray)
            }
            Spacer()
            
            /// Inline Audio Player control
            if item.type == .audio, let url = item.audioURL {
                Button(action: { audioPlayer.play(url: url) }) {
                    Image(systemName: (audioPlayer.isPlaying && audioPlayer.currentlyPlayingURL == url) ? "pause.circle.fill" : "play.circle.fill")
                        .font(.system(size: 28))
                        .foregroundColor(.brandPrimary)
                }
                .buttonStyle(BorderlessButtonStyle())
            }
            
            Button(action: { showDetail = true }) {
                Image(systemName: "info.circle").font(.system(size: 22)).foregroundColor(.gray)
            }
            .buttonStyle(BorderlessButtonStyle())
        }
        .padding(.vertical, 4)
        .sheet(isPresented: $showDetail) {
            BirdDetailView(bird: item.bird, customImage: item.image, location: (item.latitude != nil && item.longitude != nil) ? (item.latitude!, item.longitude!) : nil)
        }
    }
    
    func formatDate(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "de_DE")
        formatter.dateFormat = "d. MMM, HH:mm"
        return formatter.string(from: date)
    }
}

// MARK: - Real-time Audio Processing View
struct AudioScannerView: View {
    @ObservedObject var viewModel: AudioInputManager
    @ObservedObject var audioPlayer: AudioPlayer
    @State private var selectedBird: Bird?
    
    var body: some View {
        ZStack {
            Color.backgroundSubtle.edgesIgnoringSafeArea(.all)
            VStack(spacing: 0) {
                Text(viewModel.isRecording ? "Live Scanner" : "Scanner")
                    .font(.cardTitle).foregroundColor(.brandPrimary).padding(.vertical)
                
                // MARK: - Animated Scan Indicator
                ZStack {
                    Circle()
                        .fill(viewModel.isRecording ? Color.brandAccent.opacity(0.2) : Color.gray.opacity(0.1))
                        .frame(width: Design.Size.scannerCircle, height: Design.Size.scannerCircle)
                        .scaleEffect(viewModel.isRecording ? 1.2 : 1.0)
                        .animation(viewModel.isRecording ? Animation.easeInOut(duration: 1.0).repeatForever(autoreverses: true) : .default, value: viewModel.isRecording)
                    
                    if let heroBird = viewModel.latestBird, viewModel.isRecording {
                        BirdImageView(bird: heroBird, size: Design.Size.birdHeroImage)
                            .shadow(radius: 5)
                            .onTapGesture { selectedBird = heroBird }
                    } else {
                        Image(systemName: "mic.circle.fill")
                            .font(.system(size: Design.Size.birdHeroImage))
                            .foregroundColor(.brandPrimary)
                    }
                }
                .frame(height: 280)
                
                if let heroBird = viewModel.latestBird, viewModel.isRecording {
                    Text(heroBird.germanName).font(.title3.bold()).foregroundColor(.textDark)
                        .padding(.top, 8).padding(.bottom, 20)
                        .onTapGesture { selectedBird = heroBird }
                } else {
                    Text(" ").font(.title3).padding(.top, 8).padding(.bottom, 20)
                }
                
                // MARK: - Visual Feedback (Spectrogram)
                if let spec = viewModel.latestSpectrogram {
                    VStack {
                        Text("Live Spectrogram").font(.caption2).foregroundColor(.textGray)
                        Image(uiImage: spec).resizable().interpolation(.none).scaledToFit().frame(height: 80).background(Color.black).border(Color.textGray, width: 1)
                    }.padding(.bottom)
                }
                
                // MARK: - Session Statistics
                List {
                    ForEach(viewModel.sessionStats) { stat in
                        LiveSessionRow(stat: stat, audioPlayer: audioPlayer)
                            .listRowBackground(Color.white)
                            .listRowSeparator(.visible)
                    }
                }
                .listStyle(.plain)
                .scrollContentBackground(.hidden)
                
                // MARK: - Controls
                Button(action: { withAnimation { viewModel.toggleScanning() } }) {
                    HStack {
                        Image(systemName: viewModel.isRecording ? "stop.fill" : "play.fill")
                        Text(viewModel.isRecording ? "Stoppen" : "Starten")
                    }
                    .foregroundColor(.white).padding().background(viewModel.isRecording ? Color.brandAccent : Color.brandPrimary).cornerRadius(Design.Radius.xLarge)
                }
                .padding(.bottom)
            }
        }
        .sheet(item: $selectedBird) { bird in BirdDetailView(bird: bird) }
    }
}

// MARK: - Specialized Components
struct LiveSessionRow: View {
    let stat: BirdStatistic
    @ObservedObject var audioPlayer: AudioPlayer
    @State private var showDetail = false
    var body: some View {
        HStack(spacing: 12) {
            BirdImageCircle(bird: stat.bird, size: 42)
            VStack(alignment: .leading, spacing: 2) {
                Text(stat.bird.germanName).font(.system(size: 16, weight: .semibold)).foregroundColor(.textDark)
                Text("\(stat.count) Mal erkannt").font(.caption).foregroundColor(.textGray)
            }
            Spacer()
            if let url = stat.latestAudioURL {
                Button(action: { audioPlayer.play(url: url) }) {
                    Image(systemName: (audioPlayer.isPlaying && audioPlayer.currentlyPlayingURL == url) ? "pause.circle.fill" : "play.circle.fill")
                        .font(.system(size: 28)).foregroundColor(.brandPrimary)
                }
                .buttonStyle(BorderlessButtonStyle())
            }
            Button(action: { showDetail = true }) {
                Image(systemName: "info.circle").font(.system(size: 22)).foregroundColor(.textGray)
            }
            .buttonStyle(BorderlessButtonStyle())
        }
        .padding(.vertical, 4)
        .sheet(isPresented: $showDetail) { BirdDetailView(bird: stat.bird) }
    }
}

struct OptionCard: View {
    let title: String; let icon: String; let color: Color; let subtitle: String
    var body: some View {
        ZStack(alignment: .bottomLeading) {
            color.opacity(0.9)
            VStack(alignment: .leading) {
                Image(systemName: icon).font(.title2).foregroundColor(.white)
                Spacer()
                Text(title).font(.cardTitle).foregroundColor(.white)
                Text(subtitle).font(.caption).foregroundColor(.white.opacity(0.8))
            }
            .padding()
        }
        .cornerRadius(Design.Radius.card)
    }
}

struct BirdImageView: View {
    let bird: Bird; let size: CGFloat
    var body: some View {
        if let uiImage = UIImage(named: bird.scientificName) {
            Image(uiImage: uiImage).resizable().scaledToFill().frame(width: size, height: size).clipShape(Circle())
        } else {
            Circle().fill(Color.gray.opacity(0.2)).frame(width: size, height: size)
        }
    }
}

struct BirdImageCircle: View {
    let bird: Bird; var userImage: UIImage? = nil; let size: CGFloat
    var body: some View {
        if let uImg = userImage {
            Image(uiImage: uImg).resizable().scaledToFill().frame(width: size, height: size).clipShape(Circle()).overlay(Circle().stroke(Color.white, lineWidth: 1))
        } else if let assetImg = UIImage(named: bird.scientificName) {
            Image(uiImage: assetImg).resizable().scaledToFill().frame(width: size, height: size).clipShape(Circle())
        } else {
            Circle().fill(Color.gray.opacity(0.3)).frame(width: size, height: size).overlay(Image(systemName: "camera.fill").font(.system(size: size * 0.5)).foregroundColor(.white))
        }
    }
}

// MARK: - Detailed Encyclopedia View
struct BirdDetailView: View {
    let bird: Bird
    var customImage: UIImage? = nil
    var location: (lat: Double, long: Double)? = nil
    
    var body: some View {
        GeometryReader { geometry in
            ScrollView(.vertical, showsIndicators: false) {
                VStack(alignment: .leading, spacing: 0) {
                    
                    // MARK: - Header Asset (User Photo vs. Default)
                    if let userImg = customImage {
                        Image(uiImage: userImg)
                            .resizable()
                            .scaledToFill()
                            .frame(width: geometry.size.width, height: 300)
                            .clipped()
                    } else if let uiImage = UIImage(named: bird.scientificName) {
                        Image(uiImage: uiImage)
                            .resizable()
                            .scaledToFill()
                            .frame(width: geometry.size.width, height: 300)
                            .clipped()
                    } else {
                        Rectangle()
                            .fill(Color(UIColor.systemGray6))
                            .frame(width: geometry.size.width, height: 300)
                    }
                    
                    // MARK: - Content Section
                    VStack(alignment: .leading, spacing: 24) {
                        VStack(alignment: .leading, spacing: 8) {
                            Text(bird.germanName)
                                .font(.largeTitle)
                                .bold()
                                .foregroundColor(.black)
                                .fixedSize(horizontal: false, vertical: true)
                            
                            Text(bird.scientificName)
                                .font(.title3)
                                .italic()
                                .foregroundColor(.gray)
                        }
                        
                        Divider()
                        
                        VStack(alignment: .leading, spacing: 20) {
                            DetailRow(icon: "tag.fill", label: "Kategorie", value: bird.category)
                            DetailRow(icon: "leaf.fill", label: "Lebensraum", value: bird.habitat)
                            DetailRow(icon: "chart.bar.fill", label: "Vorkommen", value: bird.occurrence)
                            
                            // MARK: - Map Integration
                            if let loc = location {
                                HStack(alignment: .top, spacing: 16) {
                                    Image(systemName: "location.fill")
                                        .resizable()
                                        .scaledToFit()
                                        .frame(width: 20, height: 20)
                                        .foregroundColor(.blue)
                                        .padding(.top, 3)
                                    
                                    VStack(alignment: .leading, spacing: 4) {
                                        Text("FUNDORT")
                                            .font(.caption)
                                            .fontWeight(.bold)
                                            .foregroundColor(.gray)
                                            .textCase(.uppercase)
                                        
                                        Button(action: {
                                            /// URL Scheme for Apple Maps navigation
                                            let url = URL(string: "http://maps.apple.com/?ll=\(loc.lat),\(loc.long)")!
                                            UIApplication.shared.open(url)
                                        }) {
                                            HStack {
                                                Text("Auf Karte zeigen")
                                                    .font(.body)
                                                    .bold()
                                                Image(systemName: "arrow.up.right.circle")
                                            }
                                            .foregroundColor(.blue)
                                        }
                                    }
                                }
                            }
                        }
                        
                        Divider()
                        
                        // MARK: - External Link Integration
                        Link(destination: bird.wikiURL) {
                            HStack {
                                Spacer()
                                Image(systemName: "book.fill")
                                Text("Auf Wikipedia lesen")
                                Spacer()
                            }
                            .font(.headline)
                            .foregroundColor(.white)
                            .padding()
                            .background(Color.blue)
                            .cornerRadius(12)
                        }
                        .padding(.top, 10)
                        
                        Spacer(minLength: 50)
                    }
                    .padding(24)
                    .frame(width: geometry.size.width)
                    .background(Color.white)
                }
            }
            .background(Color.white)
            .edgesIgnoringSafeArea(.top)
        }
        .navigationTitle(bird.germanName)
        .navigationBarTitleDisplayMode(.inline)
    }
}

// MARK: - Reusable UI Primitives
struct DetailRow: View {
    let icon: String; let label: String; let value: String
    var body: some View {
        HStack(alignment: .top, spacing: 16) {
            Image(systemName: icon).resizable().scaledToFit().frame(width: 20, height: 20).foregroundColor(.blue).padding(.top, 3)
            VStack(alignment: .leading, spacing: 4) {
                Text(label).font(.caption).fontWeight(.bold).foregroundColor(.gray).textCase(.uppercase)
                Text(value).font(.body).foregroundColor(.black).fixedSize(horizontal: false, vertical: true)
            }
            Spacer()
        }
    }
}

struct InfoRow: View {
    let icon: String; let title: String; let value: String
    var body: some View {
        HStack {
            Image(systemName: icon).foregroundColor(.brandPrimary).frame(width: 30)
            VStack(alignment: .leading) {
                Text(title).font(.caption).foregroundColor(.textGray).textCase(.uppercase)
                Text(value).font(.body).foregroundColor(.textDark)
            }
        }
    }
}

#Preview {
    ContentView()
}
