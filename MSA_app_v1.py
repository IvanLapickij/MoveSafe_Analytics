
#reference video https://blog.roboflow.com/getting-started-with-roboflow/
#reference documentation https://docs.roboflow.com/workflows/deploy-a-workflow
#reference Roboflow football project video https://www.youtube.com/watch?v=aBVGKoNZQUw&t
#Check for devices(camera)
#ffmpeg -list_devices true -f dshow -i dummy


#1.Run Docker server clever_noether

#2.Run comman to project camera view to pipeline using terminal
#ffmpeg -f dshow -rtbufsize 256M -i video="Integrated Camera" -preset ultrafast -tune zerolatency -f flv rtmp://192.168.0.170:1935/stream

#3. run python appclication
# Import the InferencePipeline object
from inference import InferencePipeline
from inference.core.interfaces.stream.sinks import render_boxes

#Video samples
RTMP_STREAM_URL = "videos/bundesliga1.mp4"

#Lenovo
# RTMP_STREAM_URL = "rtmp://127.0.0.1:1935/stream"

#MSI
# RTMP_STREAM_URL = "rtmp://192.168.0.170:1935/stream"

#do something with the predictions of each frame
def my_sink(result, video_frame):
    
    # Visualize detections on the video stream
    render_boxes(result, video_frame)
    print(result) # prints result of each frame
    

# initialize a pipeline object
pipeline = InferencePipeline.init(
    api_key="tvZVhjN9hMWkURbVo84w",
    model_id="football-players-detection-3zvbc/12",  # Model ID
    video_reference=RTMP_STREAM_URL, # Path to video, RSTP stream, device id (int, usually 0 for built in webcams), or RTSP stream url
    on_prediction=my_sink
)
pipeline.start() #start the pipeline
pipeline.join() #wait for the pipeline thread to finish


