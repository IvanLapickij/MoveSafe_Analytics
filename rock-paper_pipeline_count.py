import cv2
from inference import InferencePipeline
from inference.core.interfaces.stream.sinks import render_boxes

# Define the RTMP stream URL
RTMP_STREAM_URL = "rtmp://192.168.0.170:1935/stream"

# Define a custom `on_prediction` function
def custom_on_prediction(pipeline, data):
    if hasattr(data, "predictions"):  # Ensure predictions are available
        predictions = data.predictions  # Access predictions
        for prediction in predictions:
            detected_class = prediction.get("class")
            if detected_class == "Paper":  # Check if the detected class is "Paper"
                print("It's a Paper!")
            else: print('ups')

    # Optionally, pass the data to render_boxes for visualization
    render_boxes(pipeline, data)

# Create an inference pipeline object
pipeline = InferencePipeline.init(
    model_id="rock-paper-scissors-sxsw/14",  # Model ID
    video_reference=RTMP_STREAM_URL,  # RTMP stream as the video source
    on_prediction=custom_on_prediction,  # Use the custom function for predictions
    api_key="tvZVhjN9hMWkURbVo84w",  # Roboflow API key
)

# Start the pipeline
pipeline.start()

# Wait for the pipeline to finish
pipeline.join()
