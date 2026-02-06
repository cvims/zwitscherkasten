# Zwitscherkasten

**ZwitscherkastenVision** is a small toolkit for object detection including classification of bird species. 

---

## Features

- Download image data from iNaturalist dataset (`DataProcessing`)
- Weak supervised data preprocessing for getting bounding boxes and image crops on a classical image classification dataset assuming only one bird species per image. (`DataProcessing/`)
- Train custom Yolo26 for detection and classification of 253 bird species (`Training/`)
- Train image classification with MobileNetV3, EfficientNetB0 and EfficientNetB3 on crops (`Training/`)
- Basic inference pipelines. Needs to be adapted to the target hardware (`Inference/`)
- Utilities for exporting  models to ONNX. (`Tools/`)

---

## Repository Structure

- `DataProcessing/` — scripts to search, filter, and download audio files; plot mel-spectrograms.
  - `prepre_crops_with_yolov11.py` — Extract bird crops from downloaded images using YOLOv11.
  - `download_inat_images.py` — Download iNaturalist images for bird species classification.
  - `data_split.py` — Performs 60/20/20 stratified split per species on cropped images from prepare_crops_with_yolov11.py.
  - `metadata.csv` — metadata for download
  - `prepare_yolo_dataset.py` — preparing dataset for End-to-End YOLO training


- `Training/` — training code and dataset preparation per model family.
  - `Yolo26s_253classes/`, `Yolo26n_253classes/`, `Yolo26m_253classes/`
  - Each contains `Train_Bird_Classification_Yolo26*.py` script.
  - `Species_Classifier/`
  - contains `train_species_classifier.py` and `run_testset.py`

- `Tools/` — small utilities
  - `yolo_to_onnx.py` — export trained yolo models to ONNX.

- `Inference/` — Inference skript
  - `VisionSplitPipeline/` - Two-stage pipeline 
  - `VisionE2EPipeline/` - Multi-Class Object Detection with custom Yolo
  - each contains `vision_pipeline_config.json`, `classes.txt`, `vision_split_pipeline.py`


---
## Downloading the Data from iNaturalist

```bash
python DataProcessing/download_inat_images.py
```
## Training end-to-end pipeline (multi-class object detection)

1. Prepare data:

```bash
python DataProcessing/prepare_yolo_dataset.py
```

2. Train a model (Yolo):

```bash
python Training/Yolo26n_253classes/Train_Bird_Classification_Yolo26n.py
```

## Training two-staged classification pipeline

1. Prepare data:
   
```bash
python DataProcessing/prepre_crops_with_yolov11.py
```
```bash
python DataProcessing/data_split.py
```

2. Train a model (e.g. EfficientNet-B1):

```bash
python Training/Species_classifier/train_species_classifier.py --model effnet_b1
```

3. Inference on Testset:

```bash
python Training/Species_classifier/run_testset.py --model-dir runs-cls/effnet_b1{Timestamp}
```
