import cv2
import matplotlib.pyplot as plt
from inference import InferencePipeline
from inference.core.interfaces.stream.sinks import render_boxes

# Define the RTMP stream URL
RTMP_STREAM_URL = "rtmp://192.168.0.170:1935/stream"

# Create an inference pipeline object
pipeline = InferencePipeline.init(
    model_id="rock-paper-scissors-sxsw/14",  # Model ID
    video_reference=RTMP_STREAM_URL,  # RTMP stream as the video source
    on_prediction=render_boxes,  # Render inference results
    api_key="tvZVhjN9hMWkURbVo84w",  # Roboflow API key
)

# Start the pipeline
pipeline.start()

# Wait for the pipeline to finish
pipeline.join()


