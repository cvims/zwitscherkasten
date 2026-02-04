//
//  ImageClassifier.swift
//  ZwitscherkastenApp
//
//  Created by Elias Häring on 13.01.26.
//

import CoreML
import UIKit
import VideoToolbox
import Accelerate

class ImageClassifier {
    
    private var mlModel: MLModel?
    private let inputName: String = "images"
    
    // --- CONFIGURATION (Must match Python training script) ---
    private var inputWidth: Int = 224
    private var inputHeight: Int = 224
    
    // Confidence threshold: 55%
    private let confidenceThreshold: Float = 0.2
    
    // Normalization: ImageNet Mean/Std
    private let normMean: [Float] = [0.48637559726934554, 0.4840848466125252, 0.4481748922135937]
    private let normStd: [Float]  = [0.22855992423587554, 0.22867489680861877, 0.25331295050563984]
    
    // --- CORRECTED LABEL LIST ---
    // Must correspond exactly to the order in classes.txt
    private let labels = [
        "Acanthis_flammea",
        "Acridotheres_tristis",
        "Acrocephalus_arundinaceus",
        "Acrocephalus_dumetorum",
        "Acrocephalus_palustris",
        "Acrocephalus_schoenobaenus",
        "Acrocephalus_scirpaceus",
        "Actitis_hypoleucos",
        "Aegithalos_caudatus",
        "Aegolius_funereus",
        "Alauda_arvensis",
        "Alcedo_atthis",
        "Alectoris_rufa",
        "Alopochen_aegyptiaca",
        "Anarhynchus_alexandrinus",
        "Anas_crecca",
        "Anas_platyrhynchos",
        "Anser_albifrons",
        "Anser_anser",
        "Anser_brachyrhynchus",
        "Anthus_campestris",
        "Anthus_cervinus",
        "Anthus_hodgsoni",
        "Anthus_petrosus",
        "Anthus_pratensis",
        "Anthus_richardi",
        "Anthus_spinoletta",
        "Anthus_trivialis",
        "Apus_apus",
        "Apus_pallidus",
        "Ardea_alba",
        "Ardea_cinerea",
        "Ardea_ibis",
        "Ardea_purpurea",
        "Arenaria_interpres",
        "Asio_flammeus",
        "Asio_otus",
        "Astur_gentilis",
        "Athene_noctua",
        "Bombycilla_garrulus",
        "Botaurus_minutus",
        "Botaurus_stellaris",
        "Branta_bernicla",
        "Branta_canadensis",
        "Branta_leucopsis",
        "Bubo_bubo",
        "Bubo_virginianus",
        "Bucephala_clangula",
        "Burhinus_oedicnemus",
        "Buteo_buteo",
        "Calandrella_brachydactyla",
        "Calcarius_lapponicus",
        "Calidris_alba",
        "Calidris_alpina",
        "Calliope_calliope",
        "Caprimulgus_europaeus",
        "Carduelis_carduelis",
        "Carpodacus_erythrinus",
        "Catharus_guttatus",
        "Catharus_ustulatus",
        "Certhia_brachydactyla",
        "Certhia_familiaris",
        "Cettia_cetti",
        "Charadrius_hiaticula",
        "Chlidonias_hybrida",
        "Chloris_chloris",
        "Chroicocephalus_ridibundus",
        "Circus_aeruginosus",
        "Cisticola_juncidis",
        "Coccothraustes_coccothraustes",
        "Coloeus_monedula",
        "Columba_oenas",
        "Columba_palumbus",
        "Corvus_corax",
        "Corvus_cornix",
        "Corvus_corone",
        "Corvus_frugilegus",
        "Coturnix_coturnix",
        "Crex_crex",
        "Cuculus_canorus",
        "Cyanistes_caeruleus",
        "Cygnus_cygnus",
        "Cygnus_olor",
        "Delichon_urbicum",
        "Dendrocopos_leucotos",
        "Dendrocopos_major",
        "Dendrocoptes_medius",
        "Dryobates_minor",
        "Dryocopus_martius",
        "Dumetella_carolinensis",
        "Egretta_garzetta",
        "Emberiza_calandra",
        "Emberiza_cia",
        "Emberiza_cirlus",
        "Emberiza_citrinella",
        "Emberiza_hortulana",
        "Emberiza_pusilla",
        "Emberiza_rustica",
        "Emberiza_schoeniclus",
        "Eremophila_alpestris",
        "Erithacus_rubecula",
        "Eudromias_morinellus",
        "Falco_peregrinus",
        "Falco_tinnunculus",
        "Ficedula_albicollis",
        "Ficedula_hypoleuca",
        "Ficedula_parva",
        "Fringilla_coelebs",
        "Fringilla_montifringilla",
        "Fulica_atra",
        "Galerida_cristata",
        "Gallinago_gallinago",
        "Gallinula_chloropus",
        "Garrulus_glandarius",
        "Glaucidium_passerinum",
        "Grus_grus",
        "Haematopus_ostralegus",
        "Himantopus_himantopus",
        "Hippolais_icterina",
        "Hippolais_polyglotta",
        "Hirundo_rustica",
        "Hydroprogne_caspia",
        "Ichthyaetus_melanocephalus",
        "Jynx_torquilla",
        "Lanius_collurio",
        "Lanius_excubitor",
        "Larus_argentatus",
        "Larus_canus",
        "Larus_fuscus",
        "Larus_marinus",
        "Larus_michahellis",
        "Limosa_lapponica",
        "Limosa_limosa",
        "Linaria_cannabina",
        "Locustella_fluviatilis",
        "Locustella_luscinioides",
        "Locustella_naevia",
        "Lophophanes_cristatus",
        "Loxia_curvirostra",
        "Loxia_pytyopsittacus",
        "Lullula_arborea",
        "Luscinia_luscinia",
        "Luscinia_megarhynchos",
        "Luscinia_svecica",
        "Mareca_penelope",
        "Mareca_strepera",
        "Melanitta_nigra",
        "Melanocorypha_calandra",
        "Melospiza_melodia",
        "Merops_apiaster",
        "Molothrus_ater",
        "Motacilla_alba",
        "Motacilla_cinerea",
        "Motacilla_citreola",
        "Motacilla_flava",
        "Muscicapa_striata",
        "Nucifraga_caryocatactes",
        "Numenius_arquata",
        "Numenius_phaeopus",
        "Nycticorax_nycticorax",
        "Oenanthe_oenanthe",
        "Oriolus_oriolus",
        "Otus_scops",
        "Pandion_haliaetus",
        "Panurus_biarmicus",
        "Parus_major",
        "Passer_domesticus",
        "Passer_montanus",
        "Passerina_cyanea",
        "Perdix_perdix",
        "Periparus_ater",
        "Petronia_petronia",
        "Phasianus_colchicus",
        "Pheucticus_ludovicianus",
        "Phoenicopterus_roseus",
        "Phoenicurus_ochruros",
        "Phoenicurus_phoenicurus",
        "Phylloscopus_bonelli",
        "Phylloscopus_borealis",
        "Phylloscopus_collybita",
        "Phylloscopus_fuscatus",
        "Phylloscopus_humei",
        "Phylloscopus_ibericus",
        "Phylloscopus_inornatus",
        "Phylloscopus_proregulus",
        "Phylloscopus_sibilatrix",
        "Phylloscopus_trochiloides",
        "Phylloscopus_trochilus",
        "Pica_pica",
        "Picoides_tridactylus",
        "Picus_canus",
        "Picus_viridis",
        "Plectrophenax_nivalis",
        "Pluvialis_apricaria",
        "Pluvialis_squatarola",
        "Podiceps_cristatus",
        "Poecile_montanus",
        "Poecile_palustris",
        "Porzana_porzana",
        "Prunella_modularis",
        "Psittacula_krameri",
        "Puffinus_puffinus",
        "Pyrrhocorax_pyrrhocorax",
        "Pyrrhula_pyrrhula",
        "Rallus_aquaticus",
        "Recurvirostra_avosetta",
        "Regulus_ignicapilla",
        "Regulus_regulus",
        "Remiz_pendulinus",
        "Riparia_riparia",
        "Saxicola_rubetra",
        "Saxicola_rubicola",
        "Scolopax_rusticola",
        "Serinus_serinus",
        "Setophaga_americana",
        "Sitta_europaea",
        "Spinus_spinus",
        "Sterna_hirundo",
        "Sterna_paradisaea",
        "Sternula_albifrons",
        "Streptopelia_decaocto",
        "Streptopelia_turtur",
        "Strix_aluco",
        "Strix_uralensis",
        "Sturnus_vulgaris",
        "Sylvia_atricapilla",
        "Sylvia_borin",
        "Tachybaptus_ruficollis",
        "Tadorna_tadorna",
        "Tarsiger_cyanurus",
        "Tetrastes_bonasia",
        "Thalasseus_sandvicensis",
        "Thinornis_dubius",
        "Tringa_erythropus",
        "Tringa_glareola",
        "Tringa_nebularia",
        "Tringa_ochropus",
        "Tringa_totanus",
        "Troglodytes_troglodytes",
        "Turdus_iliacus",
        "Turdus_merula",
        "Turdus_migratorius",
        "Turdus_philomelos",
        "Turdus_pilaris",
        "Turdus_torquatus",
        "Turdus_viscivorus",
        "Tyto_alba",
        "Upupa_epops",
        "Vanellus_vanellus",
        "Vireo_olivaceus",
        "Zapornia_pusilla",
        "Zonotrichia_albicollis",
        "Zonotrichia_leucophrys"
    ]
    
    // --- INIT ---
    init() {
        print("\n --- START: ImageClassifier Init (Corrected Labels) ---")
        
        // Attempt to find the model file in the bundle
        var modelURL = Bundle.main.url(forResource: "VisualClassifier", withExtension: "mlmodelc") ??
                       Bundle.main.url(forResource: "VisualClassifier", withExtension: "mlpackage") ??
                       Bundle.main.url(forResource: "VisualClassifier", withExtension: "mlmodel")
        
        // Fallback: Search for any compiled model with "Visual" in the name
        if modelURL == nil {
            let allC = Bundle.main.urls(forResourcesWithExtension: "mlmodelc", subdirectory: nil) ?? []
            if let best = allC.first(where: { $0.lastPathComponent.contains("Visual") }) { modelURL = best }
        }
        
        guard let finalURL = modelURL else {
            print(" ERROR: Model file not found."); return
        }
        
        do {
            // Compile model if necessary (for .mlmodel)
            let modelToLoad = (finalURL.pathExtension == "mlmodelc") ? finalURL : try MLModel.compileModel(at: finalURL)
            self.mlModel = try MLModel(contentsOf: modelToLoad)
            
            // Validate input dimensions based on model metadata (Default: 224x224)
            if let inputDesc = self.mlModel?.modelDescription.inputDescriptionsByName["images"] {
                if let multi = inputDesc.multiArrayConstraint {
                    let shape = multi.shape.map { $0.intValue }
                    if shape.count >= 2 {
                        let dims = shape.filter { $0 > 50 }
                        if dims.count >= 2 {
                            self.inputWidth = dims[0]
                            self.inputHeight = dims[1]
                        }
                    }
                }
            }
            print(" VisualClassifier loaded. Expected input: \(inputWidth)x\(inputHeight)")
            print("  Labels Count: \(labels.count)")
            
        } catch {
            print(" Error initializing model: \(error)")
        }
    }
    
    // --- CLASSIFY ---
    func classify(image: UIImage, completion: @escaping (String?, Double) -> Void) {
        guard let model = self.mlModel else { completion(nil, 0.0); return }
        
        // Offload heavy processing to a background thread
        DispatchQueue.global(qos: .userInitiated).async {
            
            // Preprocessing (Center Crop + Resize + Mean/Std)
            guard let multiArray = image.preprocessWithConfig(
                width: self.inputWidth,
                height: self.inputHeight,
                mean: self.normMean,
                std: self.normStd
            ) else {
                DispatchQueue.main.async { completion(nil, 0.0) }
                return
            }
            
            do {
                // Prepare input and execute prediction
                let inputValue = MLFeatureValue(multiArray: multiArray)
                let provider = try MLDictionaryFeatureProvider(dictionary: [self.inputName: inputValue])
                let output = try model.prediction(from: provider)
                
                // Extract output logits
                var foundMultiArray: MLMultiArray?
                for name in output.featureNames {
                    if let val = output.featureValue(for: name), let arr = val.multiArrayValue {
                        foundMultiArray = arr
                        break
                    }
                }
                
                if let logits = foundMultiArray {
                    let probabilities = self.softmax(logits)
                    
                    // --- TOP 5 LOGGING ---
                    let top5 = self.findTopIndices(in: probabilities, count: 5)
                    print("\n --- ANALYSIS RESULT (TOP 5) ---")
                    for (rank, (idx, prob)) in top5.enumerated() {
                        if idx < self.labels.count {
                            let name = self.labels[idx]
                            let pct = String(format: "%.2f %%", prob * 100)
                            print("\(rank+1). \(name): \(pct)")
                        }
                    }
                    print("------------------------------------\n")
                    
                    let (maxIndex, maxValue) = top5.first ?? (-1, -1.0)
                    
                    // Return result on Main Thread
                    DispatchQueue.main.async {
                        // Check against confidence threshold (0.55)
                        if maxValue >= self.confidenceThreshold {
                            if maxIndex >= 0 && maxIndex < self.labels.count {
                                let name = self.labels[maxIndex]
                                let percentage = String(format: "%.2f %%", maxValue * 100)
                                print(" MATCH for App: \(name) (\(percentage))")
                                completion(name, Double(maxValue))
                            } else {
                                completion(nil, 0.0)
                            }
                        } else {
                            let percentage = String(format: "%.2f %%", maxValue * 100)
                            print(" Uncertain: \(percentage) (Required: \(self.confidenceThreshold * 100)%)")
                            completion(nil, 0.0)
                        }
                    }
                }
            } catch {
                print(" CoreML Error: \(error)")
                DispatchQueue.main.async { completion(nil, 0.0) }
            }
        }
    }
    
    // --- HELPER ---
    private func softmax(_ multiArray: MLMultiArray) -> [Float] {
        let count = multiArray.count
        let ptr = UnsafeBufferPointer<Float32>(start: multiArray.dataPointer.bindMemory(to: Float32.self, capacity: count), count: count)
        let logits = Array(ptr)
        let maxLogit = logits.max() ?? 0
        let exps = logits.map { exp($0 - maxLogit) }
        let sumExps = exps.reduce(0, +)
        return exps.map { $0 / sumExps }
    }
    
    private func findTopIndices(in values: [Float], count: Int) -> [(Int, Float)] {
        let indexedValues = values.enumerated().map { ($0, $1) }
        let sorted = indexedValues.sorted { $0.1 > $1.1 }
        return Array(sorted.prefix(count))
    }
}

// MARK: - EXTENSION: PREPROCESSING (Mean/Std + Crop)
extension UIImage {
    func preprocessWithConfig(width: Int, height: Int, mean: [Float], std: [Float]) -> MLMultiArray? {
        let sideLength = min(self.size.width, self.size.height)
        
        // Center Crop Calculation
        let cropRect = CGRect(
            x: (self.size.width - sideLength) / 2.0,
            y: (self.size.height - sideLength) / 2.0,
            width: sideLength,
            height: sideLength
        )
        
        UIGraphicsBeginImageContextWithOptions(CGSize(width: width, height: height), true, 1.0)
        guard let ctx = UIGraphicsGetCurrentContext() else { return nil }
        
        // Perform Crop & Resize
        if let cgImage = self.cgImage?.cropping(to: cropRect) {
            let drawRect = CGRect(x: 0, y: 0, width: width, height: height)
            ctx.translateBy(x: 0, y: CGFloat(height))
            ctx.scaleBy(x: 1.0, y: -1.0)
            ctx.draw(cgImage, in: drawRect)
        }
        
        let newImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        
        guard let finalCgImage = newImage?.cgImage else { return nil }
        
        // Extract Raw Bytes
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width
        var rawBytes = [UInt8](repeating: 0, count: width * height * 4)
        
        let context = CGContext(data: &rawBytes,
                                width: width,
                                height: height,
                                bitsPerComponent: 8,
                                bytesPerRow: bytesPerRow,
                                space: CGColorSpaceCreateDeviceRGB(),
                                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue)
        
        context?.draw(finalCgImage, in: CGRect(x: 0, y: 0, width: CGFloat(width), height: CGFloat(height)))
        
        // Normalize and fill MultiArray
        guard let array = try? MLMultiArray(shape: [1, 3, NSNumber(value: height), NSNumber(value: width)], dataType: .float32) else {
            return nil
        }
        
        let ptr = array.dataPointer.bindMemory(to: Float32.self, capacity: width * height * 3)
        let count = width * height
        
        for i in 0..<count {
            let pixelIndex = i * 4
            let r_raw = Float32(rawBytes[pixelIndex])
            let g_raw = Float32(rawBytes[pixelIndex+1])
            let b_raw = Float32(rawBytes[pixelIndex+2])
            
            // Apply Normalization (Value = (Raw/255 - Mean) / Std)
            let r_norm = ((r_raw / 255.0) - mean[0]) / std[0]
            let g_norm = ((g_raw / 255.0) - mean[1]) / std[1]
            let b_norm = ((b_raw / 255.0) - mean[2]) / std[2]
            
            ptr[i]           = r_norm
            ptr[count + i]   = g_norm
            ptr[count*2 + i] = b_norm
        }
        
        return array
    }
}
