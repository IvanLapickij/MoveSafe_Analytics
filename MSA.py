import os
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import time
import threading
import numpy as np
from ultralytics import YOLO  # For YOLO Pose
import torch
from transformers import AutoProcessor, SiglipVisionModel
from more_itertools import chunked
import umap
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime
# For Roboflow football model inference (using inference-gpu)
from inference import get_model
import os
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import time
import threading
import numpy as np
from ultralytics import YOLO  # For YOLO Pose
import torch
from transformers import AutoProcessor, SiglipVisionModel
from more_itertools import chunked
import umap
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime
# For Roboflow football model inference (using inference-gpu)
from inference import get_model

# --- Roboflow Model Setup ---
ROBOFLOW_API_KEY = "tvZVhjN9hMWkURbVo84w"
PLAYER_DETECTION_MODEL_ID = "movesafep4/3"  # Use your model ID
PLAYER_DETECTION_MODEL = get_model(model_id=PLAYER_DETECTION_MODEL_ID, api_key=ROBOFLOW_API_KEY)
ROBOFLOW_API_KEY_FIELD = ROBOFLOW_API_KEY
FIELD_DETECTION_MODEL_ID = "msa_keypoint_detection/1"
FIELD_DETECTION_MODEL = get_model(model_id=FIELD_DETECTION_MODEL_ID, api_key=ROBOFLOW_API_KEY_FIELD)

# --- Import supervision for annotation ---
try:
    import supervision as sv
except ImportError:
    raise ImportError("Please install the supervision package (pip install supervision)")

# Setup annotators for the football model.
box_annotator = sv.BoxAnnotator(
    color=sv.ColorPalette.from_hex(['#FF8C00', '#00BFFF', '#FF1493', '#FFD700']),
    thickness=2
)
# A separate label annotator for football (if needed)
football_label_annotator = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
    text_color=sv.Color.from_hex('#000000'),
    text_position=sv.Position.BOTTOM_CENTER
)
ellipse_annotator = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
    thickness=2
)
triangle_annotator = sv.TriangleAnnotator(
    color=sv.Color.from_hex('#FFD700'),
    base=25,
    height=21,
    outline_thickness=1
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

# Football Detection globals
football_thread = None
football_model = None  # Will hold the loaded football model

# Recording globals
recording = False
video_writer = None

# Clustering globals
BATCH_SIZE = 32
EMBEDDINGS_MODEL = SiglipVisionModel.from_pretrained('google/siglip-base-patch16-224').to('cuda' if torch.cuda.is_available() else 'cpu')
EMBEDDINGS_PROCESSOR = AutoProcessor.from_pretrained('google/siglip-base-patch16-224')
REDUCER = umap.UMAP(n_components=3)
CLUSTERING_MODEL = KMeans(n_clusters=2)

# FPS control (default 60 FPS)
root = tk.Tk()
fps_value = tk.IntVar(root, value=60)

# --- Additional globals for Football Tracking and Annotation ---
BALL_ID = 0  # Assumed ball class ID
tracker = sv.ByteTrack()  # Initialize ByteTrack tracker
tracker.reset()

# --- Global Variables for Collision Graph ---
current_distance = None
collision_events = []   # List to record collision event times (seconds since start)
collision_state = False # Flag to avoid repeated logging during continuous collision
start_time = time.time()  # Global start time reference

# --- Matplotlib Figures for Graphs ---
# Create a figure for the distance graph
distance_figure = plt.Figure(figsize=(5,2), dpi=100)
distance_canvas = None  # To be created after the frame is set up

# Create a figure for the collision graph
collision_figure = plt.Figure(figsize=(5,2), dpi=100)
collision_canvas = None

# --- Inference Helper Functions ---
def football_inference():
    global latest_frame, annotated_frame, model_active, football_model, tracker, current_distance
    global collision_events, collision_state, start_time
    frame_counter = 0
    process_every = 1
    scale_factor = 0.5
    while model_active:
        if latest_frame is not None:
            frame_counter += 1
            frame = latest_frame.copy()
            if frame_counter % process_every == 0:
                try:
                    original_h, original_w = frame.shape[:2]
                    small_frame = cv2.resize(frame, (int(original_w * scale_factor), int(original_h * scale_factor)))
                    result = football_model.infer(small_frame, confidence=0.3)[0]
                    detections = sv.Detections.from_inference(result)
                    # Scale back the coordinates to the original frame size
                    detections.xyxy = detections.xyxy / scale_factor
                    
                    # Gather positions for red and blue players.
                    red_positions = []
                    blue_positions = []
                    for class_name, bbox in zip(detections['class_name'], detections.xyxy):
                        if class_name == "Player Red":
                            x1, y1, x2, y2 = bbox
                            red_positions.append(((x1+x2)/2.0, (y1+y2)/2.0))
                            print(f"Detected {class_name} at x: {(x1+x2)/2.0:.2f}, y: {(y1+y2)/2.0:.2f}")
                        elif class_name == "Player Blue":
                            x1, y1, x2, y2 = bbox
                            blue_positions.append(((x1+x2)/2.0, (y1+y2)/2.0))
                            print(f"Detected {class_name} at x: {(x1+x2)/2.0:.2f}, y: {(y1+y2)/2.0:.2f}")
                    
                    # Compute average positions for each team and calculate distance if both exist.
                    if red_positions and blue_positions:
                        avg_red = np.mean(red_positions, axis=0)
                        avg_blue = np.mean(blue_positions, axis=0)
                        current_distance = np.linalg.norm(np.array(avg_red) - np.array(avg_blue))
                    else:
                        current_distance = None

                    # Check for collision: if distance is less than 50 pixels.
                    if current_distance is not None and current_distance < 50:
                        if not collision_state:
                            # Record collision event time (in seconds since start)
                            collision_events.append(time.time() - start_time)
                            collision_state = True
                    else:
                        collision_state = False
                    
                    # Continue with your existing detection processing.
                    ball_detections = detections[detections.class_id == BALL_ID]
                    ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)
                    all_detections = detections[detections.class_id != BALL_ID]
                    all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
                    all_detections.class_id -= 1
                    tracked_detections = tracker.update_with_detections(detections=all_detections)
                    labels = [f"#{tracker_id}" for tracker_id in tracked_detections.tracker_id]
                    annotated = ellipse_annotator.annotate(scene=frame.copy(), detections=tracked_detections)
                    annotated = football_label_annotator.annotate(scene=annotated, detections=tracked_detections, labels=labels)
                    annotated = triangle_annotator.annotate(scene=annotated, detections=ball_detections)
                    annotated_frame = annotated
                except Exception as e:
                    print(f"[ERROR] Football inference error: {e}")
        time.sleep(0.03)

# --- Graph Update Functions ---
def update_distance_graph():
    global current_distance, distance_canvas, distance_figure
    distance_figure.clf()
    ax = distance_figure.add_subplot(111)
    if current_distance is not None:
        # Plot a single bar indicating the distance (in pixels)
        ax.bar(['Distance'], [current_distance], color='blue')
        ax.set_ylim(0, max(200, current_distance + 20))
        ax.set_ylabel("Distance (pixels)")
        ax.set_title("Distance between Teams")
    else:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", fontsize=12)
    distance_canvas.draw()
    # Schedule next graph update after 100ms
    root.after(100, update_distance_graph)

def update_collision_graph():
    global collision_events, collision_canvas, collision_figure
    collision_figure.clf()
    ax = collision_figure.add_subplot(111)
    if collision_events:
        # Plot collision events as scatter points along the time axis.
        ax.scatter(collision_events, [1]*len(collision_events), color='red')
        ax.set_ylim(0, 2)
        ax.set_xlabel("Time (s)")
        ax.set_yticks([])
        ax.set_title("Collision Events (Distance < 50 px)")
    else:
        ax.text(0.5, 0.5, "No collisions", ha="center", va="center", fontsize=12)
    collision_canvas.draw()
    # Schedule next graph update after 100ms
    root.after(100, update_collision_graph)

# --- Button Command Functions and Other GUI Functions ---
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"output_{timestamp}.avi"
    video_writer = cv2.VideoWriter(filename, fourcc, 10, (width, height))
    if not video_writer.isOpened():
        messagebox.showerror("Recording Error", "Failed to open video writer.")
        return
    recording = True
    print(f"[INFO] Recording started. Saving to {filename}")

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

def show_stream():
    global stop_rtmp_flag, rtmp_thread
    print("[INFO] 'Show Stream' button clicked.")
    rtmp_url = url_combo.get()
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
    global model_active, active_model, yolo_model, yolo_thread, football_thread, football_model
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
    elif chosen_model == "Players detection":
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
    global model_active, active_model, yolo_thread, football_thread, annotated_frame
    print("[INFO] Stop Model clicked.")
    model_active = False
    active_model = None
    if yolo_thread and yolo_thread.is_alive():
        yolo_thread.join()
    if football_thread and football_thread.is_alive():
        football_thread.join()
    annotated_frame = None

# Global variable to track time between frames
last_time = time.time()
def update_frame():
    global latest_frame, video_writer, recording, model_active, active_model, annotated_frame, last_time
    if latest_frame is None:
        video_label.config(text="No Video", image="")
    else:
        try:
            current_time = time.time()
            delta = current_time - last_time
            fps = 1.0 / delta if delta > 0 else 0
            last_time = current_time

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
            
            cv2.putText(frame_to_show, f"FPS: {fps:.2f}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                        (255, 255, 255), 2, cv2.LINE_AA)
            
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
    root.after(10, update_frame)

def get_video_files():
    video_folder = "videos"
    if not os.path.exists(video_folder):
        return []
    video_files = [f for f in os.listdir(video_folder) 
                   if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    return video_files

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

# Left control panel
left_frame = ttk.Frame(root, width=300)
left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

input_frame = ttk.Frame(left_frame, padding=10)
input_frame.pack(fill=tk.X)
url_label = ttk.Label(input_frame, text="VIDEO URL or File:")
url_label.grid(row=0, column=0, sticky=tk.W, pady=(0,5))
url_combo = ttk.Combobox(input_frame, state="normal", width=40)
url_combo.grid(row=1, column=0, padx=(0,5), pady=(0,10))

def update_url_combo():
    video_files = [os.path.join("videos", f) for f in os.listdir("videos") 
                   if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))] if os.path.exists("videos") else []
    url_combo['values'] = video_files
    if video_files:
        url_combo.set(video_files[0])
    else:
        url_combo.set("")
update_url_combo()

stream_button_frame = ttk.Frame(left_frame)
stream_button_frame.pack(pady=10)
btn_show_stream_ctrl = ttk.Button(stream_button_frame, text="Show Stream", command=on_show_stream)
btn_show_stream_ctrl.grid(row=0, column=0, padx=5)
btn_stop_stream = ttk.Button(stream_button_frame, text="Stop Stream", command=on_stop_stream)
btn_stop_stream.grid(row=0, column=1, padx=5)

model_label = ttk.Label(left_frame, text="Choose Model:")
model_label.pack(pady=(20, 5))
model_options = ["Players detection", "YOLO Pose"]
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

# Right frame for video and graphs
right_frame = ttk.Frame(root)
right_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

top_frame = ttk.Frame(right_frame)
top_frame.pack(side=tk.TOP, expand=True, fill=tk.BOTH)
video_label = tk.Label(top_frame, text="No Stream", bg="black", fg="white")
video_label.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

# Bottom frame divided into collision graph (top) and distance graph (bottom)
bottom_frame = ttk.Frame(right_frame, height=400)
bottom_frame.pack(side=tk.BOTTOM, fill=tk.BOTH)

collision_frame = ttk.Frame(bottom_frame)
collision_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
distance_frame = ttk.Frame(bottom_frame)
distance_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# Embed collision graph in collision_frame
collision_canvas = FigureCanvasTkAgg(collision_figure, master=collision_frame)
collision_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
collision_label = ttk.Label(collision_frame, text="Collision Graph", background="black", foreground="white")
collision_label.pack(padx=10, pady=10)

# Embed distance graph in distance_frame
distance_canvas = FigureCanvasTkAgg(distance_figure, master=distance_frame)
distance_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
distance_label = ttk.Label(distance_frame, text="Distance Graph", background="black", foreground="white")
distance_label.pack(padx=10, pady=10)

# Start updating graphs
update_collision_graph()
update_distance_graph()

update_frame()
root.protocol("WM_DELETE_WINDOW", lambda: [root.destroy()])
root.mainloop()
