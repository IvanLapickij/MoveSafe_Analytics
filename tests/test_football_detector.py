import numpy as np
from inference_models.football_detector import FootballDetector
from collision_tracker import CollisionTracker

class DummyModel:
    def infer(self, image, confidence=0.3):
        return [{
            "predictions": [],
            "class_name": [],
            "class_id": [],
            "x": [], "y": [], "width": [], "height": [],
            "confidence": [],
        }]

def test_football_detector_runs():
    dummy_model = DummyModel()
    tracker = CollisionTracker()
    detector = FootballDetector(dummy_model, tracker)

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result_frame = detector.predict(dummy_frame)
    assert result_frame is not None
