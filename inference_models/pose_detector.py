# inference_models/pose_detector.py
import cv2
from ultralytics import YOLO

class PoseDetector:
    def __init__(self, model=None):
        if model is None:
            # Create a default YOLO model if none is provided.
            self.model = YOLO("yolo11n-pose.pt")
        else:
            self.model = model

    def predict(self, frame):
        try:
            results = self.model.predict(frame, conf=0.5)
            return results[0].plot()
        except Exception as e:
            print(f"[YOLO Pose Error] {e}")
            return frame

