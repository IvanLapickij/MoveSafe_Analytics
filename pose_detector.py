class PoseDetector:
    def __init__(self, model):
        self.model = model

    def predict(self, frame):
        try:
            results = self.model.predict(frame, conf=0.5, device="cuda")  # 🔥 force to CUDA
            return results[0].plot()
        except Exception as e:
            print(f"[YOLO Pose Error] {e}")
            return frame

