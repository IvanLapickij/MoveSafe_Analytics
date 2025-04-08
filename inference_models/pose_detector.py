# inference_models/pose_detector.py
from ultralytics import YOLO

class PoseDetector:
    def __init__(self, model_path="yolo11n-pose.pt", confidence=0.5):
        self.model = YOLO(model_path)
        self.confidence = confidence

    def predict(self, frame):
        try:
            results = self.model.predict(frame, conf=self.confidence)
            return results[0].plot()
        except Exception as e:
            print(f"[ERROR] YOLO Pose inference error: {e}")
            return frame