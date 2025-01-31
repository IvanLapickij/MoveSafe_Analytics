import cv2
from inference import InferencePipeline
from inference.core.interfaces.stream.sinks import render_boxes

# Define the RTMP stream URL
RTMP_STREAM_URL = "rtmp://127.0.0.1:1935/stream"

# Create an inference pipeline object
pipeline = InferencePipeline.init(
    model_id="football-players-detection-3zvbc/12",  # Model ID
    video_reference=RTMP_STREAM_URL,  # RTMP stream as the video source
    api_key="tvZVhjN9hMWkURbVo84w",  # Roboflow API key
)

# Start the pipeline
pipeline.start()

# Wait for the pipeline to finish
pipeline.join()
