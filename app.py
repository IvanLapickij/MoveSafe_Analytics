import cv2
import math
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Replace with your YOLO model path if custom

# Open video capture
cap = cv2.VideoCapture('vid0.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)  # Get frames per second of the video

# Initialize variables for tracking
tracker = cv2.legacy.TrackerCSRT_create()
prev_coords = None  # For speed calculation
metrics = []  # To store player metrics

# Read the first frame and detect objects
ret, frame = cap.read()
if ret:
    results = model(frame)  # YOLO object detection
    # Display detections and allow user to choose a player to track
    for detection in results[0].boxes:  # Access detections from YOLOv8
        x1, y1, x2, y2 = detection.xyxy[0]  # Bounding box coordinates
        confidence = detection.conf[0]      # Confidence score
        class_id = int(detection.cls[0])    # Class ID

        # Convert to integer for drawing
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # Draw bounding boxes for detected persons
        if class_id == 0:  # Assuming '0' corresponds to players
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Player {confidence:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Let the user select a player to track
cv2.imshow('Select Player', frame)
bbox = cv2.selectROI('Select Player', frame, False)

# Validate the bbox
if bbox == (0, 0, 0, 0):  # User pressed 'c' or canceled the selection
    print("No valid ROI selected. Exiting...")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# Initialize the tracker with the selected ROI
tracker.init(frame, bbox)

# Process video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Update tracker
    success, bbox = tracker.update(frame)
    if success:
        x, y, w, h = map(int, bbox)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Calculate speed
        new_coords = (x + w // 2, y + h // 2)
        if prev_coords:
            distance = math.sqrt((new_coords[0] - prev_coords[0])**2 +
                                 (new_coords[1] - prev_coords[1])**2)
            speed = distance * fps  # Speed in pixels per second
            metrics.append({'player_id': 1, 'speed': speed})  # Assuming player_id = 1
            cv2.putText(frame, f"Speed: {speed:.2f} pixels/sec", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        prev_coords = new_coords

    cv2.imshow('Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save metrics to CSV
df = pd.DataFrame(metrics)
df.to_csv('football_metrics.csv', index=False)

# Visualize player speeds
speeds = [m['speed'] for m in metrics]
plt.hist(speeds, bins=5, alpha=0.7)
plt.title('Player Speeds')
plt.xlabel('Speed (pixels/sec)')
plt.ylabel('Frequency')
plt.show()
