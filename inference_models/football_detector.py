import cv2
import numpy as np
import supervision as sv


class FootballDetector:
    def __init__(self, model, tracker, collision_tracker=None, scale_factor=0.5):
        self.model = model
        self.tracker = tracker
        self.collision_tracker = collision_tracker
        self.scale_factor = scale_factor

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
        try:
            BALL_ID = 0
            original_h, original_w = frame.shape[:2]
            small_frame = cv2.resize(
                frame, (int(original_w * self.scale_factor), int(original_h * self.scale_factor))
            )

            # Inference call
            inference_result = self.model.infer(small_frame, confidence=0.3)

            # Extract result (handle tuple or list from Roboflow)
            result = inference_result[0] if isinstance(inference_result, (tuple, list)) else inference_result

            # If result is a dict with "predictions"
            if isinstance(result, dict) and "predictions" in result:
                result = result["predictions"]

            # Convert to supervision detections
            detections = sv.Detections.from_inference(result)
            detections.xyxy = detections.xyxy / self.scale_factor  # scale back to original resolution

            # Separate and annotate ball vs players
            ball_detections = detections[detections.class_id == BALL_ID]
            ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

            player_detections = detections[detections.class_id != BALL_ID]
            player_detections = player_detections.with_nms(threshold=0.5, class_agnostic=True)
            player_detections.class_id -= 1  # Adjust if needed

            # Update tracker
            tracked = self.tracker.update_with_detections(player_detections)
            labels = [f"#{id}" for id in tracked.tracker_id]

            # Update collision tracker with pairwise distance
            if self.collision_tracker and len(tracked.xyxy) >= 2:
                centers = np.array([
                    [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
                    for box in tracked.xyxy
                ])
                dists = np.linalg.norm(centers[0] - centers[1], ord=2)
                self.collision_tracker.update(dists)

            # Final annotation
            annotated = self.ellipse_annotator.annotate(frame.copy(), tracked)
            annotated = self.label_annotator.annotate(annotated, tracked, labels)
            annotated = self.triangle_annotator.annotate(annotated, ball_detections)

            return annotated

        except Exception:
            return frame  # fallback to unmodified frame if anything fails
