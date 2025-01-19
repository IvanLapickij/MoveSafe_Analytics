import cv2
import matplotlib.pyplot as plt

# Define the RTMP stream URL
RTMP_STREAM_URL = "rtmp://192.168.1.100:1935/stream"

# Open the RTMP stream
cap = cv2.VideoCapture(RTMP_STREAM_URL)

# Check if the stream is successfully opened
if not cap.isOpened():
    print("Error: Unable to open RTMP stream.")
    exit()

# Setup Matplotlib for display
plt.ion()
fig, ax = plt.subplots()

# Loop to read and display frames
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read frame from RTMP stream.")
        break

    # Convert BGR to RGB for Matplotlib
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Display the frame
    ax.imshow(frame_rgb)
    plt.pause(0.001)
    ax.clear()

# Release resources
cap.release()
plt.close()
