from ultralytics import YOLO
model = YOLO("runs/detect/253_yolo_26s/weights/best.pt")
model.export(format="onnx", imgsz=640, dynamic=False, nms=False)
