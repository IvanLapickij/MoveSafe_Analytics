import cv2
from ultralytics import YOLO

# 1. Load your local YOLO pose model
model = YOLO("yolo11n-pose.pt")

# 2. Start a generator of predictions from the RTMP source
results_generator = model.predict(source="rtmp://127.0.0.1:1935/stream", stream=True)

# 3. Loop over each prediction and display
for result in results_generator:
    # result.orig_img is the original (unannotated) frame
    # result.plot() returns an annotated numpy array with detections/keypoints drawn
    annotated_frame = result.plot()

    cv2.imshow("YOLOv8 Pose", annotated_frame)

    # Optional: press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
