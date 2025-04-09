import os
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import time
import threading
import numpy as np
from ultralytics import YOLO  # For YOLO Pose

# For Roboflow football model inference (using inference-gpu)
from inference import get_model
ROBOFLOW_API_KEY = "tvZVhjN9hMWkURbVo84w"
PLAYER_DETECTION_MODEL_ID = "movesafep4/3"  # Use your model ID


# Import supervision for annotation
try:
    import supervision as sv
except ImportError:
    raise ImportError("Please install the supervision package (pip install supervision)")

# Setup Supervision annotators for football model.
box_annotator = sv.BoxAnnotator(
    color=sv.ColorPalette.from_hex(['#FF8C00', '#00BFFF', '#FF1493', '#FFD700']),
    thickness=2
)
# We'll override the default label_annotator later for football-specific annotation.
label_annotator = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(['#FF8C00', '#00BFFF', '#FF1493', '#FFD700']),
    text_color=sv.Color.from_hex('#000000')
)

# --- Global Variables ---
stop_rtmp_flag = False
rtmp_thread = None
latest_frame = None    # Raw frame from the video source

# Inference globals
model_active = False   # True when any model inference is active
active_model = None    # Name of the selected model
annotated_frame = None # Frame after model annotations

# YOLO Pose globals
yolo_model = None
yolo_thread = None

# RPS globals
rps_thread = None

# Football Detection globals
football_thread = None
football_model = None  # Will hold the loaded football model

# Recording globals
recording = False
video_writer = None

# FPS control (default 60 FPS)
root = tk.Tk()
fps_value = tk.IntVar(root, value=60)

# --- Additional globals for Football Tracking and Annotation ---
BALL_ID = 0  # Assumed ball class ID
tracker = sv.ByteTrack()  # Initialize ByteTrack tracker
tracker.reset()

# Football-specific annotators:
ellipse_annotator = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
    thickness=2
)
# Use a label annotator with bottom-center text placement
label_annotator = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
    text_color=sv.Color.from_hex('#000000'),
    text_position=sv.Position.BOTTOM_CENTER
)
triangle_annotator = sv.TriangleAnnotator(
    color=sv.Color.from_hex('#FFD700'),
    base=25,
    height=21,
    outline_thickness=1
)

# --- Inference Helper Functions ---

def my_sink(result, video_frame):
    """
    Simulated callback for RPS.
    Draws blue boxes and prediction text.
    """
    for box in result.get("boxes", []):
        x1, y1, x2, y2 = box
        cv2.rectangle(video_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(video_frame, result.get("prediction", ""),
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    print("RPS Prediction:", result)

def rps_inference():
    """
    Simulated Rock-Paper-Scissors inference on the current frame.
    """
    global latest_frame, annotated_frame, model_active
    while model_active:
        if latest_frame is not None:
            frame = latest_frame.copy()
            try:
                result = {
                    "boxes": [(50, 50, 150, 150)],
                    "prediction": "rock"  # Could be "paper" or "scissors"
                }
                my_sink(result, frame)
                annotated_frame = frame
            except Exception as e:
                print(f"[ERROR] RPS inference error: {e}")
        time.sleep(0.03)

def yolo_inference():
    """
    Runs YOLO Pose inference on the current frame.
    """
    global latest_frame, annotated_frame, model_active, yolo_model
    while model_active:
        if latest_frame is not None:
            frame = latest_frame.copy()
            try:
                results = yolo_model.predict(frame, stream=False)
                if results:
                    result = results[0]
                    annotated = result.plot()  # Returns annotated frame
                    annotated_frame = annotated
            except Exception as e:
                print(f"[ERROR] YOLO inference error: {e}")
        time.sleep(0.03)

def football_inference():
    """
    Runs football detection inference on the current frame using the Roboflow model.
    To improve performance:
      - Only every Nth frame is processed.
      - The frame is resized (downscaled) before inference.
    After inference, detection coordinates are scaled back to the original resolution.
    Player detections (non-ball) are tracked with ByteTrack and annotated with ellipse and label annotators.
    Ball detections are padded and annotated with a triangle annotator.
    """
    global latest_frame, annotated_frame, model_active, football_model, tracker
    frame_counter = 0
    process_every = 1  # Process every 3rd frame; adjust as needed.
    scale_factor = 0.5  # Downscale factor (e.g., 0.5 means half resolution)
    
    while model_active:
        if latest_frame is not None:
            frame_counter += 1
            frame = latest_frame.copy()
            if frame_counter % process_every == 0:
                try:
                    original_h, original_w = frame.shape[:2]
                    # Resize the frame for faster inference.
                    small_frame = cv2.resize(frame, (int(original_w * scale_factor), int(original_h * scale_factor)))
                    
                    # Run inference on the downscaled frame.
                    result = football_model.infer(small_frame, confidence=0.3)[0]
                    # Convert the result into a Supervision Detections object.
                    detections = sv.Detections.from_inference(result)
                    # Scale detection coordinates back to the original resolution.
                    detections.xyxy = detections.xyxy / scale_factor
                    
                    # Separate ball detections (BALL_ID) and pad boxes.
                    ball_detections = detections[detections.class_id == BALL_ID]
                    ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)
                    
                    # Get player detections (non-ball) and apply NMS.
                    all_detections = detections[detections.class_id != BALL_ID]
                    all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
                    all_detections.class_id -= 1  # Adjust if needed
                    
                    # Update the tracker with player detections.
                    tracked_detections = tracker.update_with_detections(detections=all_detections)
                    # Generate tracker labels.
                    labels = [f"#{tracker_id}" for tracker_id in tracked_detections.tracker_id]
                    
                    # Annotate the frame:
                    annotated = ellipse_annotator.annotate(scene=frame.copy(), detections=tracked_detections)
                    annotated = label_annotator.annotate(scene=annotated, detections=tracked_detections, labels=labels)
                    annotated = triangle_annotator.annotate(scene=annotated, detections=ball_detections)
                    
                    annotated_frame = annotated
                except Exception as e:
                    print(f"[ERROR] Football inference error: {e}")
        time.sleep(0.03)

# --- Button Command Functions ---

def on_show_stream():
    show_stream()

def on_stop_stream():
    stop_rtmp()

def on_run_model():
    run_model()

def on_stop_model():
    stop_model()

def on_start_recording():
    global recording, video_writer, latest_frame
    print("[INFO] Start Recording clicked.")
    if latest_frame is None:
        messagebox.showwarning("No Video", "No video frame available to record yet.")
        return
    if recording:
        messagebox.showinfo("Recording", "Recording is already in progress.")
        return
    height, width, _ = latest_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter("output.avi", fourcc, 10, (width, height))
    if not video_writer.isOpened():
        messagebox.showerror("Recording Error", "Failed to open video writer.")
        return
    recording = True
    print("[INFO] Recording started.")

def on_stop_recording():
    global recording, video_writer
    print("[INFO] Stop Recording clicked.")
    if recording:
        recording = False
        if video_writer:
            video_writer.release()
            video_writer = None
        print("[INFO] Recording stopped.")
    else:
        messagebox.showinfo("Recording", "Recording is not in progress.")

def on_escape(event):
    root.quit()

# --- Main Functions for Stream and Model Handling ---

def show_stream():
    """
    Starts a background thread that reads frames from the video source and updates latest_frame.
    """
    global stop_rtmp_flag, rtmp_thread
    print("[INFO] 'Show Stream' button clicked.")
    rtmp_url = url_entry.get()
    if not rtmp_url:
        messagebox.showwarning("No Video URL", "Please enter a video URL.")
        return
    stop_rtmp()
    stop_rtmp_flag = False
    def rtmp_loop():
        print(f"[INFO] Video stream started with URL = {rtmp_url}")
        cap = cv2.VideoCapture(rtmp_url)
        if not cap.isOpened():
            print("[ERROR] Could not open video stream.")
            return
        while not stop_rtmp_flag:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read frame from video stream.")
                break
            global latest_frame
            latest_frame = frame
            time.sleep(0.03)
        cap.release()
        print("[INFO] Video stream ended.")
    rtmp_thread = threading.Thread(target=rtmp_loop, daemon=True)
    rtmp_thread.start()
    print("[INFO] Stream thread started.")

def stop_rtmp():
    """
    Stops the video stream thread and clears the displayed frame.
    """
    global stop_rtmp_flag, rtmp_thread, latest_frame
    print("[INFO] Stopping stream.")
    stop_rtmp_flag = True
    if rtmp_thread and rtmp_thread.is_alive():
        rtmp_thread.join()
    rtmp_thread = None
    latest_frame = None
    video_label.config(image="", text="No Video", fg="white")
    print("[INFO] Stream stopped.")

def run_model():
    """
    Starts the selected model's inference on the current video stream.
    """
    global model_active, active_model, yolo_model, yolo_thread, rps_thread, football_thread, football_model
    print("[INFO] Start Model clicked.")
    if latest_frame is None:
        messagebox.showwarning("No Stream", "Video stream is not active. Please start the stream first.")
        return
    chosen_model = model_var.get()
    active_model = chosen_model
    model_active = True
    if chosen_model == "YOLO Pose":
        try:
            yolo_model = YOLO("yolo11n-pose.pt")
        except Exception as e:
            messagebox.showerror("Model Error", f"Failed to load YOLO Pose model: {e}")
            model_active = False
            return
        yolo_thread = threading.Thread(target=yolo_inference, daemon=True)
        yolo_thread.start()
        print("[INFO] YOLO Pose inference started.")
    elif chosen_model == "rock-paper-scissors-sxsw/14":
        rps_thread = threading.Thread(target=rps_inference, daemon=True)
        rps_thread.start()
        print("[INFO] RPS inference started.")
    elif chosen_model == "movesafep4/3":
        if football_model is None:
            try:
                football_model = get_model(model_id=PLAYER_DETECTION_MODEL_ID, api_key=ROBOFLOW_API_KEY)
            except Exception as e:
                messagebox.showerror("Model Error", f"Failed to load football model: {e}")
                model_active = False
                return
        football_thread = threading.Thread(target=football_inference, daemon=True)
        football_thread.start()
        print("[INFO] Football inference started.")
    else:
        print(f"[INFO] Model overlay activated: {chosen_model}")

def stop_model():
    """
    Stops any active model inference.
    """
    global model_active, active_model, yolo_thread, rps_thread, football_thread, annotated_frame
    print("[INFO] Stop Model clicked.")
    model_active = False
    active_model = None
    if yolo_thread and yolo_thread.is_alive():
        yolo_thread.join()
    if rps_thread and rps_thread.is_alive():
        rps_thread.join()
    if football_thread and football_thread.is_alive():
        football_thread.join()
    annotated_frame = None

def update_frame():
    """
    Updates the video display with the latest frame (annotated if a model is active).
    The update interval is controlled by the FPS slider.
    """
    global latest_frame, video_writer, recording, model_active, active_model, annotated_frame
    if latest_frame is None:
        video_label.config(text="No Video", image="")
    else:
        try:
            if model_active:
                if annotated_frame is not None:
                    frame_to_show = annotated_frame.copy()
                else:
                    frame_to_show = latest_frame.copy()
                    cv2.putText(frame_to_show, f"Model: {active_model}",
                                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2, cv2.LINE_AA)
            else:
                frame_to_show = latest_frame.copy()
            frame_rgb = cv2.cvtColor(frame_to_show, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            video_label.config(image=imgtk, text="")
            video_label.image = imgtk
            if recording and video_writer is not None:
                video_writer.write(frame_to_show)
        except Exception as e:
            print("[ERROR] Could not display frame:", e)
            video_label.config(image="", text="No Video")
    current_fps = fps_value.get()
    if current_fps <= 0:
        current_fps = 60
    delay = int(1000 / current_fps)
    root.after(delay, update_frame)

# --- Build the GUI ---
root.title("Full-Screen Dark UI")
root.attributes("-fullscreen", True)

style = ttk.Style()
style.theme_use("clam")
root.configure(bg="black")
style.configure("TFrame", background="black")
style.configure("TButton", background="gray25", foreground="white",
                bordercolor="gray25", focusthickness=3, focuscolor="gray25")
style.configure("TLabel", background="black", foreground="white")
style.configure("TCombobox", fieldbackground="gray25", foreground="white")
style.configure("TEntry", fieldbackground="gray25", foreground="white")
style.map("TButton",
          foreground=[("active", "white"), ("disabled", "grey")],
          background=[("active", "gray40"), ("disabled", "gray20")])
root.bind("<Escape>", on_escape)

# Left/Right frames
left_frame = ttk.Frame(root, width=300)
left_frame.pack(side=tk.LEFT, fill=tk.Y)
right_frame = ttk.Frame(root)
right_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
top_frame = ttk.Frame(right_frame)
top_frame.pack(side=tk.TOP, expand=True, fill=tk.BOTH)
bottom_frame = ttk.Frame(right_frame, height=200)
bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)

# Left Menu Panel
url_label = ttk.Label(left_frame, text="VIDEO URL:")
url_label.pack(pady=(20, 5))
url_entry = ttk.Entry(left_frame, width=25)
# For testing, we use a local video file; change this to your RTMP URL if needed.
url_entry.insert(0, "videos/Video4_middleField_collision.mp4")
url_entry.pack(pady=(0, 20))

# FPS Slider
fps_label = ttk.Label(left_frame, text="FPS:")
fps_label.pack(pady=(10, 0))
fps_slider = tk.Scale(left_frame, from_=1, to=100, orient=tk.HORIZONTAL, variable=fps_value)
fps_slider.pack(pady=(0, 10))

stream_button_frame = ttk.Frame(left_frame)
stream_button_frame.pack(pady=10)
btn_show_stream = ttk.Button(stream_button_frame, text="Show Stream", command=on_show_stream)
btn_show_stream.grid(row=0, column=0, padx=5)
btn_stop_stream = ttk.Button(stream_button_frame, text="Stop Stream", command=on_stop_stream)
btn_stop_stream.grid(row=0, column=1, padx=5)

model_label = ttk.Label(left_frame, text="Choose Model:")
model_label.pack(pady=(20, 5))
model_options = [
    "YOLO Pose",
    "rock-paper-scissors-sxsw/14",
    "football-players-detection-3zvbc/11",
    "movesafep4/3"
]
model_var = tk.StringVar()
model_combo = ttk.Combobox(left_frame, textvariable=model_var, values=model_options, state="readonly")
model_combo.pack(pady=(0, 20))
model_combo.current(0)

model_button_frame = ttk.Frame(left_frame)
model_button_frame.pack(pady=10)
btn_run_model = ttk.Button(model_button_frame, text="Start Model", command=on_run_model)
btn_run_model.grid(row=0, column=0, padx=5)
btn_stop_model = ttk.Button(model_button_frame, text="Stop Model", command=on_stop_model)
btn_stop_model.grid(row=0, column=1, padx=5)

record_button_frame = ttk.Frame(left_frame)
record_button_frame.pack(pady=10)
btn_start_recording = ttk.Button(record_button_frame, text="Start Recording", command=on_start_recording)
btn_start_recording.grid(row=0, column=0, padx=5)
btn_stop_recording = ttk.Button(record_button_frame, text="Stop Recording", command=on_stop_recording)
btn_stop_recording.grid(row=0, column=1, padx=5)

video_label = tk.Label(top_frame, text="No Video", bg="black", fg="white")
video_label.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
bottom_label = ttk.Label(bottom_frame, text="Placeholder for future content.")
bottom_label.pack(padx=10, pady=10)

update_frame()
root.mainloop()