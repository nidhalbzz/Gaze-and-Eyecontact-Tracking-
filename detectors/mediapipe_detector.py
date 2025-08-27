import mediapipe as mp
import numpy as np

class MediaPipeFaceDetector:
    """
    Simple face detector using MediaPipe. Returns pixel bboxes [x1, y1, x2, y2].
    """

    def __init__(self, min_detection_confidence=0.5):
        self.detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=min_detection_confidence
        )

    def __call__(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        results = self.detector.process(frame_bgr[:, :, ::-1])  # BGR->RGB
        boxes = []
        if results.detections:
            for det in results.detections:
                bbox = det.location_data.relative_bounding_box
                x1 = max(int(bbox.xmin * w), 0)
                y1 = max(int(bbox.ymin * h), 0)
                x2 = min(int((bbox.xmin + bbox.width) * w), w - 1)
                y2 = min(int((bbox.ymin + bbox.height) * h), h - 1)
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
        return boxes
