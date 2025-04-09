# inference_models/pose_detector.py
import cv2
from ultralytics import YOLO

class PoseDetector:
    def __init__(self, model_path="yolo11n-pose.pt"):
        self.model = YOLO(model_path)

    def predict(self, frame):
        try:
            result = self.model.predict(frame, stream=False)[0]
            return result.plot()
        except Exception as e:
            print(f"[PoseDetector] Error: {e}")
            return frame
