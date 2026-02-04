import SwiftUI

// MARK: - 1. Colors
extension Color {
    // Brand Identity Colors
    static let brandPrimary = Color(red: 0.05, green: 0.35, blue: 0.38) // Dark Teal
    static let brandAccent = Color(red: 1.0, green: 0.45, blue: 0.35)  // Coral/Orange
    
    // Backgrounds & Text
    static let backgroundSubtle = Color(red: 0.96, green: 0.97, blue: 0.96) // Light Grey-Green
    static let textDark = Color(red: 0.1, green: 0.15, blue: 0.15)
    static let textGray = Color(red: 0.4, green: 0.45, blue: 0.45)
    
    // UI Elements
    static let shadow = Color.black.opacity(0.08)
    static let overlayDark = Color.black.opacity(0.8)
}

// MARK: - 2. Metrics
struct Design {
    
    struct Spacing {
        static let small: CGFloat = 10
        static let medium: CGFloat = 15
        static let large: CGFloat = 20
        static let xLarge: CGFloat = 30
        static let xxLarge: CGFloat = 40
        static let standardPadding: CGFloat = 20
    }
    
    struct Radius {
        static let small: CGFloat = 5
        static let medium: CGFloat = 12
        static let large: CGFloat = 20
        static let xLarge: CGFloat = 30
        static let card: CGFloat = 22
    }
    
    struct Size {
        static let optionCardHeight: CGFloat = 220
        static let scannerCircle: CGFloat = 260
        static let birdHeroImage: CGFloat = 140
        static let detailHeaderHeight: CGFloat = 350
        static let buttonHeight: CGFloat = 50
    }
    
    struct Icon {
        static let small: CGFloat = 18
        static let medium: CGFloat = 40
        static let large: CGFloat = 80
        static let huge: CGFloat = 100
    }
}

// MARK: - 3. Typography
extension Font {
    // Semantic font definitions
    
    static let heroTitle = Font.system(size: 34, weight: .heavy, design: .rounded)
    static let sectionHeader = Font.system(.headline, design: .rounded)
    static let cardTitle = Font.system(size: 24, weight: .bold, design: .rounded)
    
    static let birdNameLarge = Font.system(size: 36, weight: .heavy, design: .rounded)
    static let scientificName = Font.system(size: 18, weight: .medium, design: .serif).italic()
    
    static let bodyBold = Font.caption.weight(.bold)
    static let bodyText = Font.system(.body, design: .rounded).weight(.medium)
}

// MARK: - 4. Helper Shapes
// Custom modifier for partial corner rounding
extension View {
    func cornerRadius(_ radius: CGFloat, corners: UIRectCorner) -> some View {
        clipShape(RoundedCorner(radius: radius, corners: corners))
    }
}

struct RoundedCorner: Shape {
    var radius: CGFloat = .infinity
    var corners: UIRectCorner = .allCorners
    
    func path(in rect: CGRect) -> Path {
        let path = UIBezierPath(roundedRect: rect, byRoundingCorners: corners, cornerRadii: CGSize(width: radius, height: radius))
        return Path(path.cgPath)
    }
}
