Ultralytics® YOLOv8s outputs bounding boxes are in Q19 fixed-point format.
For GStreamer postprocessing, GSTBBoxWHScale converts these DSP-style coordinates into normalized 0–1 values.
