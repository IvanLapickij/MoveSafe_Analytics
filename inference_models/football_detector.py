import cv2
import numpy as np
import supervision as sv

class FootballDetector:
    def __init__(self, model, tracker, collision_tracker, scale_factor=0.5):
        self.model = model
        self.tracker = tracker
        self.collision_tracker = collision_tracker
        self.scale_factor = scale_factor
        self.ball_id = 0

        self.ellipse_annotator = sv.EllipseAnnotator(
            color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
            thickness=2
        )
        self.label_annotator = sv.LabelAnnotator(
            color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
            text_color=sv.Color.from_hex('#000000'),
            text_position=sv.Position.BOTTOM_CENTER
        )
        self.triangle_annotator = sv.TriangleAnnotator(
            color=sv.Color.from_hex('#FFD700'),
            base=25,
            height=21,
            outline_thickness=1
        )

    def predict(self, frame):
        h, w = frame.shape[:2]
        small_frame = cv2.resize(frame, (int(w * self.scale_factor), int(h * self.scale_factor)))

        try:
            result = self.model.infer(small_frame, confidence=0.3)[0]
        except Exception as e:
            print(f"[ERROR] Football inference error: {e}")
            return frame

        detections = sv.Detections.from_inference(result)
        detections.xyxy = detections.xyxy / self.scale_factor

        red_positions, blue_positions = [], []
        for class_name, bbox in zip(detections['class_name'], detections.xyxy):
            x1, y1, x2, y2 = bbox
            center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            if class_name == "Player Red":
                red_positions.append(center)
            elif class_name == "Player Blue":
                blue_positions.append(center)

        if red_positions and blue_positions:
            avg_red = np.mean(red_positions, axis=0)
            avg_blue = np.mean(blue_positions, axis=0)
            distance = np.linalg.norm(np.array(avg_red) - np.array(avg_blue))
        else:
            distance = None

        self.collision_tracker.update(distance)

        ball_detections = detections[detections.class_id == self.ball_id]
        ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)
        player_detections = detections[detections.class_id != self.ball_id].with_nms(threshold=0.5, class_agnostic=True)
        player_detections.class_id -= 1

        tracked = self.tracker.update_with_detections(detections=player_detections)
        labels = [f"#{id}" for id in tracked.tracker_id]

        annotated = self.ellipse_annotator.annotate(scene=frame.copy(), detections=tracked)
        annotated = self.label_annotator.annotate(scene=annotated, detections=tracked, labels=labels)
        annotated = self.triangle_annotator.annotate(scene=annotated, detections=ball_detections)

        return annotated
