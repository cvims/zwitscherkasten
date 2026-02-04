import Foundation
import UIKit
import CoreLocation

// MARK: - Data Models

// Enumeration distinguishing between audio and visual detection modes.
enum DetectionType: String, Sendable, Codable {
    case audio
    case visual
}

// Immutable data structure representing static bird species information.
struct Bird: Identifiable, Sendable, Codable {
    var id = UUID()
    let scientificName: String
    let germanName: String
    let category: String
    let occurrence: String
    let habitat: String
    
    // Playback reference URL.
    var audioURL: URL?
    
    // Generates a localized Wikipedia URL based on the scientific name.
    var wikiURL: URL {
        let cleanName = scientificName.replacingOccurrences(of: "_", with: " ")
        return URL(string: "https://de.wikipedia.org/wiki/\(cleanName.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? "")")!
    }
}

// Aggregates detection counts and percentages for session analysis.
struct BirdStatistic: Identifiable {
    let id = UUID()
    let bird: Bird
    let count: Int
    let percentage: Double
    let latestAudioURL: URL?
}

// Primary data model representing a specific detection event.
// Handles metadata persistence and file path resolution.
struct DetectionItem: Identifiable, Sendable, Codable {
    var id = UUID()
    let sessionID: UUID
    let bird: Bird
    var date: Date
    let type: DetectionType
    var audioFilename: String?
    var imageFilename: String?
    
    // GPS Coordinates
    var latitude: Double?
    var longitude: Double?
    
    // MARK: - Computed Properties
    // Resolves relative filenames to absolute URLs within the document directory.
    
    var audioURL: URL? {
        guard let name = audioFilename else { return nil }
        return FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first?.appendingPathComponent(name)
    }
    
    var image: UIImage? {
        guard let name = imageFilename,
              let url = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first?.appendingPathComponent(name),
              let data = try? Data(contentsOf: url) else { return nil }
        return UIImage(data: data)
    }
    
    // MARK: - Initialization
    // Initializes the item and handles immediate persistence of visual data to disk.
    init(sessionID: UUID, bird: Bird, date: Date, type: DetectionType, audioURL: URL? = nil, image: UIImage? = nil, location: (lat: Double, long: Double)? = nil) {
        self.sessionID = sessionID
        self.bird = bird
        self.date = date
        self.type = type
        
        // Assign GPS data
        if let loc = location {
            self.latitude = loc.lat
            self.longitude = loc.long
        }
        
        if let url = audioURL {
            self.audioFilename = url.lastPathComponent
        }
        
        if let img = image {
            let imgName = "img_\(UUID().uuidString).jpg"
            if let data = img.jpegData(compressionQuality: 0.8),
               let docDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
                try? data.write(to: docDir.appendingPathComponent(imgName))
                self.imageFilename = imgName
            }
        }
    }
    
    // MARK: - Custom Codable Implementation
    // Excludes non-serializable UIImage objects and persists only metadata.
    
    enum CodingKeys: String, CodingKey {
        case id, sessionID, bird, date, type, audioFilename, imageFilename, latitude, longitude
    }
    
    // Encodable
    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(sessionID, forKey: .sessionID)
        try container.encode(bird, forKey: .bird)
        try container.encode(date, forKey: .date)
        try container.encode(type, forKey: .type)
        try container.encode(audioFilename, forKey: .audioFilename)
        try container.encode(imageFilename, forKey: .imageFilename)
        try container.encode(latitude, forKey: .latitude)
        try container.encode(longitude, forKey: .longitude)
    }
    
    // Decodable
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(UUID.self, forKey: .id)
        sessionID = try container.decode(UUID.self, forKey: .sessionID)
        bird = try container.decode(Bird.self, forKey: .bird)
        date = try container.decode(Date.self, forKey: .date)
        type = try container.decode(DetectionType.self, forKey: .type)
        audioFilename = try container.decodeIfPresent(String.self, forKey: .audioFilename)
        imageFilename = try container.decodeIfPresent(String.self, forKey: .imageFilename)
        latitude = try container.decodeIfPresent(Double.self, forKey: .latitude)
        longitude = try container.decodeIfPresent(Double.self, forKey: .longitude)
    }
}

// Comprehensive bird species dataset based on class_map.json (256 entries).
let allBirds: [Bird] = [
    Bird(scientificName: "Acanthis_cabaret", germanName: "Alpenbirkenzeisig", category: "Finken", occurrence: "Selten", habitat: "Gebirge"),
    Bird(scientificName: "Acanthis_flammea", germanName: "Birkenzeisig", category: "Finken", occurrence: "Mittel", habitat: "Wälder"),
    Bird(scientificName: "Accipiter_gentilis", germanName: "Habicht", category: "Greifvögel", occurrence: "Mittel", habitat: "Wälder"),
    Bird(scientificName: "Acridotheres_tristis", germanName: "Hirtenmaina", category: "Stare", occurrence: "Exot", habitat: "Städte"),
    Bird(scientificName: "Acrocephalus_arundinaceus", germanName: "Drosselrohrsänger", category: "Singvögel", occurrence: "Selten", habitat: "Schilf"),
    Bird(scientificName: "Acrocephalus_dumetorum", germanName: "Buschrohrsänger", category: "Singvögel", occurrence: "Sehr Selten", habitat: "Gebüsch"),
    Bird(scientificName: "Acrocephalus_palustris", germanName: "Sumpfrohrsänger", category: "Singvögel", occurrence: "Mittel", habitat: "Feuchtgebiete"),
    Bird(scientificName: "Acrocephalus_schoenobaenus", germanName: "Schilfrohrsänger", category: "Singvögel", occurrence: "Mittel", habitat: "Schilf"),
    Bird(scientificName: "Acrocephalus_scirpaceus", germanName: "Teichrohrsänger", category: "Singvögel", occurrence: "Häufig", habitat: "Schilf"),
    Bird(scientificName: "Actitis_hypoleucos", germanName: "Flussuferläufer", category: "Watvögel", occurrence: "Mittel", habitat: "Flussufer"),
    Bird(scientificName: "Aegithalos_caudatus", germanName: "Schwanzmeise", category: "Meisen", occurrence: "Häufig", habitat: "Wälder"),
    Bird(scientificName: "Aegolius_funereus", germanName: "Raufußkauz", category: "Eulen", occurrence: "Selten", habitat: "Nadelwälder"),
    Bird(scientificName: "Alauda_arvensis", germanName: "Feldlerche", category: "Lerchen", occurrence: "Mittel", habitat: "Felder"),
    Bird(scientificName: "Alcedo_atthis", germanName: "Eisvogel", category: "Eisvögel", occurrence: "Selten", habitat: "Gewässer"),
    Bird(scientificName: "Alectoris_rufa", germanName: "Rothuhn", category: "Hühnervögel", occurrence: "Selten", habitat: "Offenland"),
    Bird(scientificName: "Alopochen_aegyptiaca", germanName: "Nilgans", category: "Entenvögel", occurrence: "Häufig", habitat: "Gewässer"),
    Bird(scientificName: "Anas_crecca", germanName: "Krickente", category: "Entenvögel", occurrence: "Mittel", habitat: "Gewässer"),
    Bird(scientificName: "Anas_platyrhynchos", germanName: "Stockente", category: "Entenvögel", occurrence: "Sehr Häufig", habitat: "Überall"),
    Bird(scientificName: "Anser_albifrons", germanName: "Blässgans", category: "Gänse", occurrence: "Wintergast", habitat: "Felder"),
    Bird(scientificName: "Anser_anser", germanName: "Graugans", category: "Gänse", occurrence: "Häufig", habitat: "Gewässer"),
    Bird(scientificName: "Anser_brachyrhynchus", germanName: "Kurzschnabelgans", category: "Gänse", occurrence: "Wintergast", habitat: "Küsten"),
    Bird(scientificName: "Anser_cygnoid_domestica", germanName: "Höckergans", category: "Gänse", occurrence: "Zuchtform", habitat: "Parks"),
    Bird(scientificName: "Anthus_campestris", germanName: "Brachpieper", category: "Pieper", occurrence: "Selten", habitat: "Offenland"),
    Bird(scientificName: "Anthus_cervinus", germanName: "Rotkehlpieper", category: "Pieper", occurrence: "Durchzügler", habitat: "Feuchtwiesen"),
    Bird(scientificName: "Anthus_hodgsoni", germanName: "Waldpieper", category: "Pieper", occurrence: "Selten", habitat: "Wälder"),
    Bird(scientificName: "Anthus_petrosus", germanName: "Strandpieper", category: "Pieper", occurrence: "Wintergast", habitat: "Küsten"),
    Bird(scientificName: "Anthus_pratensis", germanName: "Wiesenpieper", category: "Pieper", occurrence: "Mittel", habitat: "Wiesen"),
    Bird(scientificName: "Anthus_richardi", germanName: "Spornpieper", category: "Pieper", occurrence: "Selten", habitat: "Offenland"),
    Bird(scientificName: "Anthus_spinoletta", germanName: "Bergpieper", category: "Pieper", occurrence: "Mittel", habitat: "Gebirge"),
    Bird(scientificName: "Anthus_trivialis", germanName: "Baumpieper", category: "Pieper", occurrence: "Mittel", habitat: "Waldränder"),
    Bird(scientificName: "Apus_apus", germanName: "Mauersegler", category: "Segler", occurrence: "Häufig", habitat: "Städte"),
    Bird(scientificName: "Apus_pallidus", germanName: "Fahlsegler", category: "Segler", occurrence: "Selten", habitat: "Städte"),
    Bird(scientificName: "Ardea_alba", germanName: "Silberreiher", category: "Reiher", occurrence: "Mittel", habitat: "Feuchtgebiete"),
    Bird(scientificName: "Ardea_cinerea", germanName: "Graureiher", category: "Reiher", occurrence: "Häufig", habitat: "Gewässer"),
    Bird(scientificName: "Ardea_purpurea", germanName: "Purpurreiher", category: "Reiher", occurrence: "Selten", habitat: "Schilf"),
    Bird(scientificName: "Arenaria_interpres", germanName: "Steinwälzer", category: "Schnepfenvögel", occurrence: "Durchzügler", habitat: "Küsten"),
    Bird(scientificName: "Asio_flammeus", germanName: "Sumpfohreule", category: "Eulen", occurrence: "Selten", habitat: "Moore"),
    Bird(scientificName: "Asio_otus", germanName: "Waldohreule", category: "Eulen", occurrence: "Mittel", habitat: "Wälder"),
    Bird(scientificName: "Athene_noctua", germanName: "Steinkauz", category: "Eulen", occurrence: "Selten", habitat: "Streuobstwiesen"),
    Bird(scientificName: "Bombycilla_garrulus", germanName: "Seidenschwanz", category: "Seidenschwänze", occurrence: "Wintergast", habitat: "Parks"),
    Bird(scientificName: "Botaurus_stellaris", germanName: "Rohrdommel", category: "Reiher", occurrence: "Selten", habitat: "Schilf"),
    Bird(scientificName: "Branta_bernicla", germanName: "Ringelgans", category: "Gänse", occurrence: "Wintergast", habitat: "Küsten"),
    Bird(scientificName: "Branta_canadensis", germanName: "Kanadagans", category: "Gänse", occurrence: "Häufig", habitat: "Gewässer"),
    Bird(scientificName: "Branta_leucopsis", germanName: "Nonnengans", category: "Gänse", occurrence: "Mittel", habitat: "Küsten"),
    Bird(scientificName: "Bubo_bubo", germanName: "Uhu", category: "Eulen", occurrence: "Selten", habitat: "Felsen"),
    Bird(scientificName: "Bubo_virginianus", germanName: "Virginia-Uhu", category: "Eulen", occurrence: "Irrgast", habitat: "Wälder"),
    Bird(scientificName: "Bubulcus_ibis", germanName: "Kuhreiher", category: "Reiher", occurrence: "Selten", habitat: "Weiden"),
    Bird(scientificName: "Bucephala_clangula", germanName: "Schellente", category: "Entenvögel", occurrence: "Wintergast", habitat: "Gewässer"),
    Bird(scientificName: "Burhinus_oedicnemus", germanName: "Triel", category: "Triele", occurrence: "Sehr Selten", habitat: "Steppen"),
    Bird(scientificName: "Buteo_buteo", germanName: "Mäusebussard", category: "Greifvögel", occurrence: "Sehr Häufig", habitat: "Offenland"),
    Bird(scientificName: "Calandrella_brachydactyla", germanName: "Kurzehenlerche", category: "Lerchen", occurrence: "Selten", habitat: "Trockengebiete"),
    Bird(scientificName: "Calcarius_lapponicus", germanName: "Spornmonitor", category: "Ammern", occurrence: "Wintergast", habitat: "Küsten"),
    Bird(scientificName: "Calidris_alba", germanName: "Sanderling", category: "Schnepfenvögel", occurrence: "Durchzügler", habitat: "Küsten"),
    Bird(scientificName: "Calidris_alpina", germanName: "Alpenstrandläufer", category: "Schnepfenvögel", occurrence: "Durchzügler", habitat: "Küsten"),
    Bird(scientificName: "Calliope_calliope", germanName: "Rubinkehlchen", category: "Schmätzer", occurrence: "Irrgast", habitat: "Gebüsch"),
    Bird(scientificName: "Caprimulgus_europaeus", germanName: "Ziegenmelker", category: "Nachtschwalben", occurrence: "Selten", habitat: "Heiden"),
    Bird(scientificName: "Carduelis_carduelis", germanName: "Stieglitz", category: "Finken", occurrence: "Häufig", habitat: "Gärten"),
    Bird(scientificName: "Carpodacus_erythrinus", germanName: "Karmingimpel", category: "Finken", occurrence: "Selten", habitat: "Feuchtgebiete"),
    Bird(scientificName: "Catharus_guttatus", germanName: "Einsiedlerdrossel", category: "Drosseln", occurrence: "Irrgast", habitat: "Wälder"),
    Bird(scientificName: "Catharus_ustulatus", germanName: "Zwergdrossel", category: "Drosseln", occurrence: "Irrgast", habitat: "Wälder"),
    Bird(scientificName: "Certhia_brachydactyla", germanName: "Gartenbaumläufer", category: "Baumläufer", occurrence: "Häufig", habitat: "Gärten"),
    Bird(scientificName: "Certhia_familiaris", germanName: "Waldbaumläufer", category: "Baumläufer", occurrence: "Mittel", habitat: "Wälder"),
    Bird(scientificName: "Cettia_cetti", germanName: "Seidensänger", category: "Singvögel", occurrence: "Selten", habitat: "Feuchtgebiete"),
    Bird(scientificName: "Charadrius_alexandrinus", germanName: "Seeregenpfeifer", category: "Regenpfeifer", occurrence: "Selten", habitat: "Küsten"),
    Bird(scientificName: "Charadrius_dubius", germanName: "Flussregenpfeifer", category: "Regenpfeifer", occurrence: "Mittel", habitat: "Kiesflächen"),
    Bird(scientificName: "Charadrius_hiaticula", germanName: "Sandregenpfeifer", category: "Regenpfeifer", occurrence: "Mittel", habitat: "Küsten"),
    Bird(scientificName: "Charadrius_morinellus", germanName: "Mornellregenpfeifer", category: "Regenpfeifer", occurrence: "Durchzügler", habitat: "Gebirge"),
    Bird(scientificName: "Chlidonias_hybrida", germanName: "Weißbart-Seeschwalbe", category: "Seeschwalben", occurrence: "Selten", habitat: "Feuchtgebiete"),
    Bird(scientificName: "Chloris_chloris", germanName: "Grünfink", category: "Finken", occurrence: "Häufig", habitat: "Gärten"),
    Bird(scientificName: "Chroicocephalus_ridibundus", germanName: "Lachmöwe", category: "Möwen", occurrence: "Häufig", habitat: "Gewässer"),
    Bird(scientificName: "Circus_aeruginosus", germanName: "Rohrweihe", category: "Greifvögel", occurrence: "Mittel", habitat: "Schilf"),
    Bird(scientificName: "Cisticola_juncidis", germanName: "Zistensänger", category: "Singvögel", occurrence: "Selten", habitat: "Feuchtwiesen"),
    Bird(scientificName: "Coccothraustes_coccothraustes", germanName: "Kernbeißer", category: "Finken", occurrence: "Mittel", habitat: "Laubwälder"),
    Bird(scientificName: "Coloeus_monedula", germanName: "Dohle", category: "Rabenvögel", occurrence: "Häufig", habitat: "Städte"),
    Bird(scientificName: "Columba_livia_domestica", germanName: "Felsentaube", category: "Tauben", occurrence: "Häufig", habitat: "Städte"),
    Bird(scientificName: "Columba_oenas", germanName: "Hohltaube", category: "Tauben", occurrence: "Mittel", habitat: "Wälder"),
    Bird(scientificName: "Columba_palumbus", germanName: "Ringeltaube", category: "Tauben", occurrence: "Sehr Häufig", habitat: "Überall"),
    Bird(scientificName: "Corvus_corax", germanName: "Kolkrabe", category: "Rabenvögel", occurrence: "Mittel", habitat: "Wälder"),
    Bird(scientificName: "Corvus_cornix", germanName: "Nebelkrähe", category: "Rabenvögel", occurrence: "Mittel", habitat: "Offenland"),
    Bird(scientificName: "Corvus_corone", germanName: "Rabenkrähe", category: "Rabenvögel", occurrence: "Sehr Häufig", habitat: "Offenland"),
    Bird(scientificName: "Corvus_frugilegus", germanName: "Saatkrähe", category: "Rabenvögel", occurrence: "Häufig", habitat: "Felder"),
    Bird(scientificName: "Coturnix_coturnix", germanName: "Wachtel", category: "Hühnervögel", occurrence: "Selten", habitat: "Felder"),
    Bird(scientificName: "Crex_crex", germanName: "Wachtelkönig", category: "Rallen", occurrence: "Selten", habitat: "Wiesen"),
    Bird(scientificName: "Cuculus_canorus", germanName: "Kuckuck", category: "Kuckucke", occurrence: "Mittel", habitat: "Waldränder"),
    Bird(scientificName: "Cyanistes_caeruleus", germanName: "Blaumeise", category: "Meisen", occurrence: "Sehr Häufig", habitat: "Gärten"),
    Bird(scientificName: "Cygnus_cygnus", germanName: "Singschwan", category: "Schwäne", occurrence: "Wintergast", habitat: "Gewässer"),
    Bird(scientificName: "Cygnus_olor", germanName: "Höckerschwan", category: "Schwäne", occurrence: "Häufig", habitat: "Gewässer"),
    Bird(scientificName: "Delichon_urbicum", germanName: "Mehlschwalbe", category: "Schwalben", occurrence: "Häufig", habitat: "Städte"),
    Bird(scientificName: "Dendrocopos_leucotos", germanName: "Weißrückenspecht", category: "Spechte", occurrence: "Selten", habitat: "Wälder"),
    Bird(scientificName: "Dendrocopos_major", germanName: "Buntspecht", category: "Spechte", occurrence: "Häufig", habitat: "Wälder"),
    Bird(scientificName: "Dendrocoptes_medius", germanName: "Mittelspecht", category: "Spechte", occurrence: "Mittel", habitat: "Laubwälder"),
    Bird(scientificName: "Dryobates_minor", germanName: "Kleinspecht", category: "Spechte", occurrence: "Selten", habitat: "Auwälder"),
    Bird(scientificName: "Dryocopus_martius", germanName: "Schwarzspecht", category: "Spechte", occurrence: "Mittel", habitat: "Wälder"),
    Bird(scientificName: "Dumetella_carolinensis", germanName: "Katzendrossel", category: "Spottdrosseln", occurrence: "Irrgast", habitat: "Gebüsch"),
    Bird(scientificName: "Egretta_garzetta", germanName: "Seidenreiher", category: "Reiher", occurrence: "Selten", habitat: "Gewässer"),
    Bird(scientificName: "Emberiza_calandra", germanName: "Grauammer", category: "Ammern", occurrence: "Mittel", habitat: "Felder"),
    Bird(scientificName: "Emberiza_cia", germanName: "Zippammer", category: "Ammern", occurrence: "Selten", habitat: "Weinberge"),
    Bird(scientificName: "Emberiza_cirlus", germanName: "Zaunammer", category: "Ammern", occurrence: "Selten", habitat: "Weinberge"),
    Bird(scientificName: "Emberiza_citrinella", germanName: "Goldammer", category: "Ammern", occurrence: "Häufig", habitat: "Hecken"),
    Bird(scientificName: "Emberiza_hortulana", germanName: "Ortolan", category: "Ammern", occurrence: "Selten", habitat: "Felder"),
    Bird(scientificName: "Emberiza_pusilla", germanName: "Zwergammer", category: "Ammern", occurrence: "Selten", habitat: "Gebüsch"),
    Bird(scientificName: "Emberiza_rustica", germanName: "Waldammer", category: "Ammern", occurrence: "Selten", habitat: "Wälder"),
    Bird(scientificName: "Emberiza_schoeniclus", germanName: "Rohrammer", category: "Ammern", occurrence: "Mittel", habitat: "Schilf"),
    Bird(scientificName: "Eremophila_alpestris", germanName: "Ohrenlerche", category: "Lerchen", occurrence: "Wintergast", habitat: "Küsten"),
    Bird(scientificName: "Erithacus_rubecula", germanName: "Rotkehlchen", category: "Fliegenschnäpper", occurrence: "Sehr Häufig", habitat: "Gärten"),
    Bird(scientificName: "Falco_peregrinus", germanName: "Wanderfalke", category: "Falken", occurrence: "Mittel", habitat: "Felsen"),
    Bird(scientificName: "Falco_tinnunculus", germanName: "Turmfalke", category: "Falken", occurrence: "Häufig", habitat: "Offenland"),
    Bird(scientificName: "Ficedula_albicollis", germanName: "Halsbandschnäpper", category: "Fliegenschnäpper", occurrence: "Selten", habitat: "Laubwälder"),
    Bird(scientificName: "Ficedula_hypoleuca", germanName: "Trauerschnäpper", category: "Fliegenschnäpper", occurrence: "Mittel", habitat: "Wälder"),
    Bird(scientificName: "Ficedula_parva", germanName: "Zwergschnäpper", category: "Fliegenschnäpper", occurrence: "Selten", habitat: "Wälder"),
    Bird(scientificName: "Fringilla_coelebs", germanName: "Buchfink", category: "Finken", occurrence: "Sehr Häufig", habitat: "Wälder"),
    Bird(scientificName: "Fringilla_montifringilla", germanName: "Bergfink", category: "Finken", occurrence: "Wintergast", habitat: "Wälder"),
    Bird(scientificName: "Fulica_atra", germanName: "Blässhuhn", category: "Rallen", occurrence: "Häufig", habitat: "Gewässer"),
    Bird(scientificName: "Galerida_cristata", germanName: "Haubenlerche", category: "Lerchen", occurrence: "Selten", habitat: "Brachen"),
    Bird(scientificName: "Gallinago_gallinago", germanName: "Bekassine", category: "Schnepfenvögel", occurrence: "Selten", habitat: "Moore"),
    Bird(scientificName: "Gallinula_chloropus", germanName: "Teichhuhn", category: "Rallen", occurrence: "Häufig", habitat: "Gewässer"),
    Bird(scientificName: "Garrulus_glandarius", germanName: "Eichelhäher", category: "Rabenvögel", occurrence: "Häufig", habitat: "Wälder"),
    Bird(scientificName: "Glaucidium_passerinum", germanName: "Sperlingskauz", category: "Eulen", occurrence: "Selten", habitat: "Nadelwälder"),
    Bird(scientificName: "Grus_grus", germanName: "Kranich", category: "Kraniche", occurrence: "Durchzügler", habitat: "Felder"),
    Bird(scientificName: "Haematopus_ostralegus", germanName: "Austernfischer", category: "Austernfischer", occurrence: "Häufig", habitat: "Küsten"),
    Bird(scientificName: "Himantopus_himantopus", germanName: "Stelzenläufer", category: "Säbelschnäbler", occurrence: "Selten", habitat: "Feuchtgebiete"),
    Bird(scientificName: "Hippolais_icterina", germanName: "Gelbspötter", category: "Spötter", occurrence: "Mittel", habitat: "Gärten"),
    Bird(scientificName: "Hippolais_polyglotta", germanName: "Orpheusspötter", category: "Spötter", occurrence: "Selten", habitat: "Gebüsch"),
    Bird(scientificName: "Hirundo_rustica", germanName: "Rauchschwalbe", category: "Schwalben", occurrence: "Häufig", habitat: "Dörfer"),
    Bird(scientificName: "Hydroprogne_caspia", germanName: "Raubseeschwalbe", category: "Seeschwalben", occurrence: "Durchzügler", habitat: "Küsten"),
    Bird(scientificName: "Ichthyaetus_melanocephalus", germanName: "Schwarzkopfmöwe", category: "Möwen", occurrence: "Selten", habitat: "Küsten"),
    Bird(scientificName: "Ixobrychus_minutus", germanName: "Zwergdommel", category: "Reiher", occurrence: "Selten", habitat: "Schilf"),
    Bird(scientificName: "Jynx_torquilla", germanName: "Wendehals", category: "Spechte", occurrence: "Selten", habitat: "Streuobstwiesen"),
    Bird(scientificName: "Lanius_collurio", germanName: "Neuntöter", category: "Würger", occurrence: "Mittel", habitat: "Hecken"),
    Bird(scientificName: "Lanius_excubitor", germanName: "Raubwürger", category: "Würger", occurrence: "Selten", habitat: "Offenland"),
    Bird(scientificName: "Larus_argentatus", germanName: "Silbermöwe", category: "Möwen", occurrence: "Häufig", habitat: "Küsten"),
    Bird(scientificName: "Larus_canus", germanName: "Sturmmöwe", category: "Möwen", occurrence: "Häufig", habitat: "Küsten"),
    Bird(scientificName: "Larus_fuscus", germanName: "Heringsmöwe", category: "Möwen", occurrence: "Mittel", habitat: "Küsten"),
    Bird(scientificName: "Larus_marinus", germanName: "Mantelmöwe", category: "Möwen", occurrence: "Mittel", habitat: "Küsten"),
    Bird(scientificName: "Larus_michahellis", germanName: "Mittelmeermöwe", category: "Möwen", occurrence: "Mittel", habitat: "Gewässer"),
    Bird(scientificName: "Limosa_lapponica", germanName: "Pfuhlschnepfe", category: "Schnepfenvögel", occurrence: "Durchzügler", habitat: "Küsten"),
    Bird(scientificName: "Limosa_limosa", germanName: "Uferschnepfe", category: "Schnepfenvögel", occurrence: "Selten", habitat: "Feuchtwiesen"),
    Bird(scientificName: "Linaria_cannabina", germanName: "Bluthänfling", category: "Finken", occurrence: "Mittel", habitat: "Hecken"),
    Bird(scientificName: "Locustella_fluviatilis", germanName: "Schlagschwirl", category: "Schwirl", occurrence: "Selten", habitat: "Auwälder"),
    Bird(scientificName: "Locustella_luscinioides", germanName: "Rohrschwirl", category: "Schwirl", occurrence: "Selten", habitat: "Schilf"),
    Bird(scientificName: "Locustella_naevia", germanName: "Feldschwirl", category: "Schwirl", occurrence: "Mittel", habitat: "Feuchtwiesen"),
    Bird(scientificName: "Lophophanes_cristatus", germanName: "Haubenmeise", category: "Meisen", occurrence: "Mittel", habitat: "Nadelwälder"),
    Bird(scientificName: "Loxia_curvirostra", germanName: "Fichtenkreuzschnabel", category: "Finken", occurrence: "Mittel", habitat: "Nadelwälder"),
    Bird(scientificName: "Loxia_pytyopsittacus", germanName: "Kiefernkreuzschnabel", category: "Finken", occurrence: "Selten", habitat: "Nadelwälder"),
    Bird(scientificName: "Lullula_arborea", germanName: "Heidelerche", category: "Lerchen", occurrence: "Selten", habitat: "Heiden"),
    Bird(scientificName: "Luscinia_luscinia", germanName: "Sprosser", category: "Fliegenschnäpper", occurrence: "Selten", habitat: "Auwälder"),
    Bird(scientificName: "Luscinia_megarhynchos", germanName: "Nachtigall", category: "Fliegenschnäpper", occurrence: "Mittel", habitat: "Gebüsch"),
    Bird(scientificName: "Luscinia_svecica", germanName: "Blaukehlchen", category: "Fliegenschnäpper", occurrence: "Selten", habitat: "Schilf"),
    Bird(scientificName: "Mareca_penelope", germanName: "Pfeifente", category: "Entenvögel", occurrence: "Wintergast", habitat: "Gewässer"),
    Bird(scientificName: "Mareca_strepera", germanName: "Schnatterente", category: "Entenvögel", occurrence: "Mittel", habitat: "Gewässer"),
    Bird(scientificName: "Melanitta_nigra", germanName: "Trauerente", category: "Entenvögel", occurrence: "Wintergast", habitat: "Meer"),
    Bird(scientificName: "Melanocorypha_calandra", germanName: "Kalanderlerche", category: "Lerchen", occurrence: "Irrgast", habitat: "Steppen"),
    Bird(scientificName: "Melospiza_melodia", germanName: "Singammer", category: "Ammern", occurrence: "Irrgast", habitat: "Gebüsch"),
    Bird(scientificName: "Merops_apiaster", germanName: "Bienenfresser", category: "Bienenfresser", occurrence: "Selten", habitat: "Wärmegebiete"),
    Bird(scientificName: "Molothrus_ater", germanName: "Braunkopf-Kuhstärling", category: "Stärlinge", occurrence: "Irrgast", habitat: "Offenland"),
    Bird(scientificName: "Motacilla_alba", germanName: "Bachstelze", category: "Stelzen", occurrence: "Häufig", habitat: "Offenland"),
    Bird(scientificName: "Motacilla_cinerea", germanName: "Gebirgsstelze", category: "Stelzen", occurrence: "Mittel", habitat: "Bäche"),
    Bird(scientificName: "Motacilla_citreola", germanName: "Zitronenstelze", category: "Stelzen", occurrence: "Sehr Selten", habitat: "Feuchtgebiete"),
    Bird(scientificName: "Motacilla_flava", germanName: "Schafstelze", category: "Stelzen", occurrence: "Mittel", habitat: "Wiesen"),
    Bird(scientificName: "Muscicapa_striata", germanName: "Grauschnäpper", category: "Fliegenschnäpper", occurrence: "Mittel", habitat: "Gärten"),
    Bird(scientificName: "Nucifraga_caryocatactes", germanName: "Tannenhäher", category: "Rabenvögel", occurrence: "Selten", habitat: "Nadelwälder"),
    Bird(scientificName: "Numenius_arquata", germanName: "Großer Brachvogel", category: "Schnepfenvögel", occurrence: "Selten", habitat: "Feuchtwiesen"),
    Bird(scientificName: "Numenius_phaeopus", germanName: "Regenbrachvogel", category: "Schnepfenvögel", occurrence: "Durchzügler", habitat: "Küsten"),
    Bird(scientificName: "Nycticorax_nycticorax", germanName: "Nachtreiher", category: "Reiher", occurrence: "Selten", habitat: "Gewässer"),
    Bird(scientificName: "Oenanthe_oenanthe", germanName: "Steinschmätzer", category: "Schmätzer", occurrence: "Selten", habitat: "Steinig"),
    Bird(scientificName: "Oriolus_oriolus", germanName: "Pirol", category: "Pirole", occurrence: "Mittel", habitat: "Auwälder"),
    Bird(scientificName: "Otus_scops", germanName: "Zwergohreule", category: "Eulen", occurrence: "Sehr Selten", habitat: "Wärmegebiete"),
    Bird(scientificName: "Pandion_haliaetus", germanName: "Fischadler", category: "Greifvögel", occurrence: "Selten", habitat: "Seen"),
    Bird(scientificName: "Panurus_biarmicus", germanName: "Bartmeise", category: "Singvögel", occurrence: "Selten", habitat: "Schilf"),
    Bird(scientificName: "Parus_major", germanName: "Kohlmeise", category: "Meisen", occurrence: "Sehr Häufig", habitat: "Gärten"),
    Bird(scientificName: "Passer_domesticus", germanName: "Haussperling", category: "Sperlinge", occurrence: "Sehr Häufig", habitat: "Städte"),
    Bird(scientificName: "Passer_montanus", germanName: "Feldsperling", category: "Sperlinge", occurrence: "Häufig", habitat: "Felder"),
    Bird(scientificName: "Passerina_cyanea", germanName: "Indigofink", category: "Kardinäle", occurrence: "Irrgast", habitat: "Gebüsch"),
    Bird(scientificName: "Perdix_perdix", germanName: "Rebhuhn", category: "Hühnervögel", occurrence: "Selten", habitat: "Felder"),
    Bird(scientificName: "Periparus_ater", germanName: "Tannenmeise", category: "Meisen", occurrence: "Häufig", habitat: "Nadelwälder"),
    Bird(scientificName: "Petronia_petronia", germanName: "Steinsperling", category: "Sperlinge", occurrence: "Sehr Selten", habitat: "Felsen"),
    Bird(scientificName: "Phasianus_colchicus", germanName: "Fasan", category: "Hühnervögel", occurrence: "Häufig", habitat: "Felder"),
    Bird(scientificName: "Pheucticus_ludovicianus", germanName: "Rosenbrust-Kernknacker", category: "Kardinäle", occurrence: "Irrgast", habitat: "Wälder"),
    Bird(scientificName: "Phoenicopterus_roseus", germanName: "Rosaflamingo", category: "Flamingos", occurrence: "Selten", habitat: "Salzseen"),
    Bird(scientificName: "Phoenicurus_ochruros", germanName: "Hausrotschwanz", category: "Schmätzer", occurrence: "Häufig", habitat: "Dörfer"),
    Bird(scientificName: "Phoenicurus_phoenicurus", germanName: "Gartenrotschwanz", category: "Schmätzer", occurrence: "Mittel", habitat: "Gärten"),
    Bird(scientificName: "Phylloscopus_bonelli", germanName: "Berglaubsänger", category: "Laubsänger", occurrence: "Selten", habitat: "Wälder"),
    Bird(scientificName: "Phylloscopus_borealis", germanName: "Wanderlaubsänger", category: "Laubsänger", occurrence: "Irrgast", habitat: "Wälder"),
    Bird(scientificName: "Phylloscopus_collybita", germanName: "Zilpzalp", category: "Laubsänger", occurrence: "Sehr Häufig", habitat: "Wälder"),
    Bird(scientificName: "Phylloscopus_fuscatus", germanName: "Dunkellaubsänger", category: "Laubsänger", occurrence: "Irrgast", habitat: "Gebüsch"),
    Bird(scientificName: "Phylloscopus_humei", germanName: "Tienschan-Laubsänger", category: "Laubsänger", occurrence: "Irrgast", habitat: "Wälder"),
    Bird(scientificName: "Phylloscopus_ibericus", germanName: "Iberienzilpzalp", category: "Laubsänger", occurrence: "Irrgast", habitat: "Wälder"),
    Bird(scientificName: "Phylloscopus_inornatus", germanName: "Gelbbrauen-Laubsänger", category: "Laubsänger", occurrence: "Irrgast", habitat: "Wälder"),
    Bird(scientificName: "Phylloscopus_proregulus", germanName: "Goldhähnchen-Laubsänger", category: "Laubsänger", occurrence: "Irrgast", habitat: "Wälder"),
    Bird(scientificName: "Phylloscopus_sibilatrix", germanName: "Waldlaubsänger", category: "Laubsänger", occurrence: "Mittel", habitat: "Buchenwälder"),
    Bird(scientificName: "Phylloscopus_trochiloides", germanName: "Grünlaubsänger", category: "Laubsänger", occurrence: "Selten", habitat: "Wälder"),
    Bird(scientificName: "Phylloscopus_trochilus", germanName: "Fitis", category: "Laubsänger", occurrence: "Häufig", habitat: "Gebüsch"),
    Bird(scientificName: "Pica_pica", germanName: "Elster", category: "Rabenvögel", occurrence: "Häufig", habitat: "Siedlungen"),
    Bird(scientificName: "Picoides_tridactylus", germanName: "Dreizehenspecht", category: "Spechte", occurrence: "Selten", habitat: "Nadelwälder"),
    Bird(scientificName: "Picus_canus", germanName: "Grauspecht", category: "Spechte", occurrence: "Selten", habitat: "Auwälder"),
    Bird(scientificName: "Picus_viridis", germanName: "Grünspecht", category: "Spechte", occurrence: "Häufig", habitat: "Parks"),
    Bird(scientificName: "Plectrophenax_nivalis", germanName: "Schneeammer", category: "Ammern", occurrence: "Wintergast", habitat: "Küsten"),
    Bird(scientificName: "Pluvialis_apricaria", germanName: "Goldregenpfeifer", category: "Regenpfeifer", occurrence: "Durchzügler", habitat: "Felder"),
    Bird(scientificName: "Pluvialis_squatarola", germanName: "Kiebitzregenpfeifer", category: "Regenpfeifer", occurrence: "Durchzügler", habitat: "Küsten"),
    Bird(scientificName: "Podiceps_cristatus", germanName: "Haubentaucher", category: "Lappentaucher", occurrence: "Häufig", habitat: "Seen"),
    Bird(scientificName: "Poecile_montanus", germanName: "Weidenmeise", category: "Meisen", occurrence: "Mittel", habitat: "Feuchtgebiete"),
    Bird(scientificName: "Poecile_palustris", germanName: "Sumpfmeise", category: "Meisen", occurrence: "Häufig", habitat: "Laubwälder"),
    Bird(scientificName: "Porzana_porzana", germanName: "Tüpfelsumpfhuhn", category: "Rallen", occurrence: "Selten", habitat: "Sümpfe"),
    Bird(scientificName: "Prunella_modularis", germanName: "Heckenbraunelle", category: "Braunellen", occurrence: "Häufig", habitat: "Hecken"),
    Bird(scientificName: "Psittacula_krameri", germanName: "Halsbandsittich", category: "Papageien", occurrence: "Lokal Häufig", habitat: "Parks"),
    Bird(scientificName: "Puffinus_puffinus", germanName: "Schwarzschnabelsturmtaucher", category: "Sturmtaucher", occurrence: "Meer", habitat: "Meer"),
    Bird(scientificName: "Pyrrhocorax_pyrrhocorax", germanName: "Alpenkrähe", category: "Rabenvögel", occurrence: "Sehr Selten", habitat: "Gebirge"),
    Bird(scientificName: "Pyrrhula_pyrrhula", germanName: "Gimpel", category: "Finken", occurrence: "Häufig", habitat: "Gärten"),
    Bird(scientificName: "Rallus_aquaticus", germanName: "Wasserralle", category: "Rallen", occurrence: "Mittel", habitat: "Schilf"),
    Bird(scientificName: "Recurvirostra_avosetta", germanName: "Säbelschnäbler", category: "Säbelschnäbler", occurrence: "Mittel", habitat: "Küsten"),
    Bird(scientificName: "Regulus_ignicapilla", germanName: "Sommergoldhähnchen", category: "Goldhähnchen", occurrence: "Mittel", habitat: "Mischwälder"),
    Bird(scientificName: "Regulus_regulus", germanName: "Wintergoldhähnchen", category: "Goldhähnchen", occurrence: "Häufig", habitat: "Nadelwälder"),
    Bird(scientificName: "Remiz_pendulinus", germanName: "Beutelmeise", category: "Beutelmeisen", occurrence: "Selten", habitat: "Flussufer"),
    Bird(scientificName: "Riparia_riparia", germanName: "Uferschwalbe", category: "Schwalben", occurrence: "Mittel", habitat: "Steilufer"),
    Bird(scientificName: "Saxicola_rubetra", germanName: "Braunkehlchen", category: "Schmätzer", occurrence: "Selten", habitat: "Wiesen"),
    Bird(scientificName: "Saxicola_rubicola", germanName: "Schwarzkehlchen", category: "Schmätzer", occurrence: "Mittel", habitat: "Brachen"),
    Bird(scientificName: "Scolopax_rusticola", germanName: "Waldschnepfe", category: "Schnepfenvögel", occurrence: "Selten", habitat: "Wälder"),
    Bird(scientificName: "Serinus_serinus", germanName: "Girlitz", category: "Finken", occurrence: "Häufig", habitat: "Parks"),
    Bird(scientificName: "Setophaga_americana", germanName: "Meisenwaldsänger", category: "Waldsänger", occurrence: "Irrgast", habitat: "Wälder"),
    Bird(scientificName: "Sitta_europaea", germanName: "Kleiber", category: "Kleiber", occurrence: "Häufig", habitat: "Wälder"),
    Bird(scientificName: "Spinus_spinus", germanName: "Erlenzeisig", category: "Finken", occurrence: "Häufig", habitat: "Erlen"),
    Bird(scientificName: "Sterna_hirundo", germanName: "Flussseeschwalbe", category: "Seeschwalben", occurrence: "Mittel", habitat: "Gewässer"),
    Bird(scientificName: "Sterna_paradisaea", germanName: "Küstenseeschwalbe", category: "Seeschwalben", occurrence: "Mittel", habitat: "Küsten"),
    Bird(scientificName: "Sternula_albifrons", germanName: "Zwergseeschwalbe", category: "Seeschwalben", occurrence: "Selten", habitat: "Küsten"),
    Bird(scientificName: "Streptopelia_decaocto", germanName: "Türkentaube", category: "Tauben", occurrence: "Häufig", habitat: "Siedlungen"),
    Bird(scientificName: "Streptopelia_turtur", germanName: "Turteltaube", category: "Tauben", occurrence: "Selten", habitat: "Offenland"),
    Bird(scientificName: "Strix_aluco", germanName: "Waldkauz", category: "Eulen", occurrence: "Häufig", habitat: "Wälder"),
    Bird(scientificName: "Strix_uralensis", germanName: "Habichtskauz", category: "Eulen", occurrence: "Selten", habitat: "Wälder"),
    Bird(scientificName: "Sturnus_vulgaris", germanName: "Star", category: "Stare", occurrence: "Sehr Häufig", habitat: "Gärten"),
    Bird(scientificName: "Sylvia_atricapilla", germanName: "Mönchsgrasmücke", category: "Grasmücken", occurrence: "Sehr Häufig", habitat: "Gebüsch"),
    Bird(scientificName: "Sylvia_borin", germanName: "Gartengrasmücke", category: "Grasmücken", occurrence: "Mittel", habitat: "Gärten"),
    Bird(scientificName: "Tachybaptus_ruficollis", germanName: "Zwergtaucher", category: "Lappentaucher", occurrence: "Mittel", habitat: "Gewässer"),
    Bird(scientificName: "Tadorna_tadorna", germanName: "Brandgans", category: "Entenvögel", occurrence: "Häufig", habitat: "Küsten"),
    Bird(scientificName: "Tarsiger_cyanurus", germanName: "Blauschwanz", category: "Schmätzer", occurrence: "Sehr Selten", habitat: "Wälder"),
    Bird(scientificName: "Tetrastes_bonasia", germanName: "Haselhuhn", category: "Hühnervögel", occurrence: "Selten", habitat: "Wälder"),
    Bird(scientificName: "Thalasseus_sandvicensis", germanName: "Brandseeschwalbe", category: "Seeschwalben", occurrence: "Mittel", habitat: "Küsten"),
    Bird(scientificName: "Tringa_erythropus", germanName: "Dunkler Wasserläufer", category: "Schnepfenvögel", occurrence: "Durchzügler", habitat: "Gewässer"),
    Bird(scientificName: "Tringa_glareola", germanName: "Bruchwasserläufer", category: "Schnepfenvögel", occurrence: "Durchzügler", habitat: "Feuchtwiesen"),
    Bird(scientificName: "Tringa_nebularia", germanName: "Grünschenkel", category: "Schnepfenvögel", occurrence: "Durchzügler", habitat: "Gewässer"),
    Bird(scientificName: "Tringa_ochropus", germanName: "Waldwasserläufer", category: "Schnepfenvögel", occurrence: "Mittel", habitat: "Waldtümpel"),
    Bird(scientificName: "Tringa_totanus", germanName: "Rotschenkel", category: "Schnepfenvögel", occurrence: "Mittel", habitat: "Feuchtwiesen"),
    Bird(scientificName: "Troglodytes_troglodytes", germanName: "Zaunkönig", category: "Zaunkönige", occurrence: "Sehr Häufig", habitat: "Unterholz"),
    Bird(scientificName: "Turdus_iliacus", germanName: "Rotdrossel", category: "Drosseln", occurrence: "Wintergast", habitat: "Wälder"),
    Bird(scientificName: "Turdus_merula", germanName: "Amsel", category: "Drosseln", occurrence: "Sehr Häufig", habitat: "Gärten"),
    Bird(scientificName: "Turdus_migratorius", germanName: "Wanderdrossel", category: "Drosseln", occurrence: "Irrgast", habitat: "Wälder"),
    Bird(scientificName: "Turdus_philomelos", germanName: "Singdrossel", category: "Drosseln", occurrence: "Häufig", habitat: "Wälder"),
    Bird(scientificName: "Turdus_pilaris", germanName: "Wacholderdrossel", category: "Drosseln", occurrence: "Häufig", habitat: "Offenland"),
    Bird(scientificName: "Turdus_torquatus", germanName: "Ringdrossel", category: "Drosseln", occurrence: "Selten", habitat: "Gebirge"),
    Bird(scientificName: "Turdus_viscivorus", germanName: "Misteldrossel", category: "Drosseln", occurrence: "Mittel", habitat: "Wälder"),
    Bird(scientificName: "Tyto_alba", germanName: "Schleiereule", category: "Eulen", occurrence: "Selten", habitat: "Scheunen"),
    Bird(scientificName: "Upupa_epops", germanName: "Wiedehopf", category: "Wiedehopfe", occurrence: "Selten", habitat: "Weinberge"),
    Bird(scientificName: "Vanellus_vanellus", germanName: "Kiebitz", category: "Regenpfeifer", occurrence: "Mittel", habitat: "Felder"),
    Bird(scientificName: "Vireo_olivaceus", germanName: "Rotaugenvireo", category: "Vireos", occurrence: "Irrgast", habitat: "Wälder"),
    Bird(scientificName: "Zapornia_pusilla", germanName: "Zwergsumpfhuhn", category: "Rallen", occurrence: "Sehr Selten", habitat: "Sümpfe"),
    Bird(scientificName: "Zonotrichia_albicollis", germanName: "Weißkehlammer", category: "Ammern", occurrence: "Irrgast", habitat: "Gebüsch"),
    Bird(scientificName: "Zonotrichia_leucophrys", germanName: "Dachsammer", category: "Ammern", occurrence: "Irrgast", habitat: "Gebüsch")
]
