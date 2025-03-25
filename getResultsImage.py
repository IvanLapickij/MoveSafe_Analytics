from inference import get_model
ROBOFLOW_API_KEY="tvZVhjN9hMWkURbVo84w"
PLAYER_DETECTION_MODEL_ID = "movesafep4/3"
PLAYER_DETECTION_MODEL = get_model(model_id=PLAYER_DETECTION_MODEL_ID, api_key=ROBOFLOW_API_KEY)

import supervision as sv

SOURCE_VIDEO_PATH = "videos\\Video1_middleField_collision.mp4"

box_annotator = sv.BoxAnnotator(
    color=sv.ColorPalette.from_hex(['#FF8C00', '#00BFFF', '#FF1493', '#FFD700']),
    thickness=2
)
label_annotator = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(['#FF8C00', '#00BFFF', '#FF1493', '#FFD700']),
    text_color=sv.Color.from_hex('#000000')
)

frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
frame = next(frame_generator)

result = PLAYER_DETECTION_MODEL.infer(frame, confidence=0.3)[0]
detections = sv.Detections.from_inference(result)

# Print class names along with x and y coordinates.
# Here we assume detections.xywh contains a list of [x, y, width, height] for each detection.
for class_name, bbox in zip(detections['class_name'], detections.xyxy):
    x, y, width, height = bbox
    print(f"Class: {class_name}, x: {x:.2f}, y: {y:.2f}")

# Create labels for annotation (if you want to include the confidence)
labels = [
    f"{class_name} {confidence:.2f}"
    for class_name, confidence
    in zip(detections['class_name'], detections.confidence)
]

annotated_frame = frame.copy()
annotated_frame = box_annotator.annotate(
    scene=annotated_frame,
    detections=detections
)
annotated_frame = label_annotator.annotate(
    scene=annotated_frame,
    detections=detections,
    labels=labels
)

sv.plot_image(annotated_frame)
