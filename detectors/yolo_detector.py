import os
from pathlib import Path
import numpy as np

try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None

class YOLOFaceDetector:
    """
    YOLOv8 face detector. Requires a face model weights file.
    Returns pixel bboxes [x1, y1, x2, y2].
    """

    def __init__(self, weights: Path, conf: float = 0.25):
        if YOLO is None:
            raise ImportError("ultralytics not installed. pip install ultralytics")
        if not Path(weights).exists():
            raise FileNotFoundError(f"YOLO weights not found at {weights}")
        self.model = YOLO(str(weights))
        self.conf = conf

    def __call__(self, frame_bgr):
        res = self.model.predict(source=frame_bgr, conf=self.conf, verbose=False)[0]
        boxes = []
        if res.boxes is not None and len(res.boxes) > 0:
            for b in res.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = b[:4].astype(int).tolist()
                boxes.append([x1, y1, x2, y2])
        return boxes
