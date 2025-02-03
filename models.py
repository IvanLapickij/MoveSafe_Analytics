import tkinter as tk
from tkinter import ttk, messagebox
import threading
import cv2
import numpy as np
from PIL import Image, ImageTk
import time

# Roboflow Inference references:
from inference import InferencePipeline
from inference.core.interfaces.stream.sinks import render_boxes

#############################
# GUI + Pipeline Framework #
#############################

# GLOBALS for recording
recording = False
video_writer = None

# We'll store the latest frame with detections rendered
latest_frame = None
stop_pipeline_flag = False

def on_prediction(result, video_frame):
    """
    Callback from Roboflow's pipeline that receives predictions 
    and the raw video_frame (OpenCV image).
    - Renders bounding boxes on the video_frame
    - Stores the processed frame globally so the GUI can display it
    - Writes to disk if 'recording' is True
    """
    global latest_frame, recording, video_writer

    # Draw bounding boxes on the frame
    render_boxes(result, video_frame)

    # Keep track of the latest processed frame (BGR -> store as is)
    latest_frame = video_frame

    # If recording is toggled on, write the processed frame
    if recording and video_writer is not None:
        video_writer.write(video_frame)


def start_pipeline(model_id, rtmp_url):
    """
    Initialize and start the Roboflow InferencePipeline.
    This runs in a background thread so the GUI remains responsive.
    """
    global stop_pipeline_flag

    stop_pipeline_flag = False  # reset flag
    try:
        pipeline = InferencePipeline.init(
            api_key="tvZVhjN9hMWkURbVo84w",   # <--- Replace with your real API key
            model_id=model_id,
            video_reference=rtmp_url,
            on_prediction=on_prediction
        )
        pipeline.start()

        # Keep the pipeline alive until stop_pipeline_flag is set
        while not stop_pipeline_flag:
            time.sleep(0.1)

        # Once flagged to stop, shut down pipeline
        pipeline.stop()
    except Exception as e:
        messagebox.showerror("Error", f"Failed to start pipeline:\n{e}")

def stop_pipeline():
    """
    Signals the background thread to stop the pipeline.
    """
    global stop_pipeline_flag
    stop_pipeline_flag = True

def toggle_record():
    """
    Toggles the recording state.
    If record is OFF -> ON, create a VideoWriter.
    If record is ON -> OFF, release the VideoWriter.
    """
    global recording, video_writer

    if not recording:
        # Start recording
        recording = True
        filename = time.strftime("recording_%Y%m%d-%H%M%S.mp4")

        # For simplicity, assume RTMP feed is (width=1280, height=720).
        # If different, detect actual w/h from the first frame or pipeline’s metadata.
        frame_width = 1280
        frame_height = 720
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(filename, fourcc, 10, (frame_width, frame_height))
        record_button.config(text="Stop Recording")
    else:
        # Stop recording
        recording = False
        if video_writer is not None:
            video_writer.release()
            video_writer = None
        record_button.config(text="Start Recording")

#############################
# Tkinter Main Application #
#############################

def update_frame():
    """
    Periodically called to update the GUI’s preview Label with the latest frame.
    """
    if latest_frame is not None:
        # Convert the latest frame from BGR to RGB
        frame_rgb = cv2.cvtColor(latest_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.config(image=imgtk)
        video_label.image = imgtk  # Keep a reference to avoid garbage collection

    # Schedule the next update
    root.after(50, update_frame)  # ~20 FPS update in the GUI

def run_model():
    """
    Called when 'Run Model' is clicked. Spawns a thread to start the pipeline.
    """
    selected_model = model_var.get()
    if not selected_model:
        messagebox.showwarning("No Model", "Please select a model first.")
        return

    rtmp_url = rtmp_entry.get()
    if not rtmp_url:
        messagebox.showwarning("No RTMP URL", "Please enter an RTMP URL.")
        return

    # Start the pipeline in a background thread
    pipeline_thread = threading.Thread(target=start_pipeline, args=(selected_model, rtmp_url), daemon=True)
    pipeline_thread.start()

def stop_model():
    """
    Called when 'Stop Model' is clicked. Tells the pipeline thread to exit.
    """
    stop_pipeline()

###########
#  Setup  #
###########
root = tk.Tk()
root.title("Model Selector")

# Dropdown for choosing model
model_var = tk.StringVar()
model_options = [
    "rock-paper-scissors-sxsw/14",
    "football-players-detection-3zvbc/12",
    "some-other-model-id/1"
]
model_label = ttk.Label(root, text="Choose Model:")
model_label.pack(pady=(10, 0))

model_menu = ttk.Combobox(root, textvariable=model_var, values=model_options, state="readonly")
model_menu.pack()

# Entry for RTMP URL
rtmp_label = ttk.Label(root, text="RTMP URL:")
rtmp_label.pack(pady=(10, 0))

rtmp_entry = ttk.Entry(root, width=40)
rtmp_entry.insert(0, "rtmp://127.0.0.1:1935/stream")  # Default placeholder
rtmp_entry.pack(pady=(0, 10))

# Buttons for controlling pipeline and recording
button_frame = tk.Frame(root)
button_frame.pack()

run_button = ttk.Button(button_frame, text="Run Model", command=run_model)
run_button.grid(row=0, column=0, padx=5)

stop_button = ttk.Button(button_frame, text="Stop Model", command=stop_model)
stop_button.grid(row=0, column=1, padx=5)

global record_button
record_button = ttk.Button(button_frame, text="Start Recording", command=toggle_record)
record_button.grid(row=0, column=2, padx=5)

# A Label widget to display the incoming video frames
video_label = tk.Label(root, text="Live Inference Video")
video_label.pack(pady=10)

# Start periodic frame updates
update_frame()

# Start Tkinter loop
root.mainloop()
