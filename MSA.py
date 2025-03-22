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

# For Roboflow football model inference (using inference-gpu)
from inference import get_model
ROBOFLOW_API_KEY = "tvZVhjN9hMWkURbVo84w"
PLAYER_DETECTION_MODEL_ID = "football-players-detection-3zvbc/11"
PLAYER_DETECTION_MODEL = get_model(model_id=PLAYER_DETECTION_MODEL_ID, api_key=ROBOFLOW_API_KEY)

ROBOFLOW_API_KEY_FIELD = ROBOFLOW_API_KEY
FIELD_DETECTION_MODEL_ID = "football-field-detection-f07vi/14"
FIELD_DETECTION_MODEL = get_model(model_id=FIELD_DETECTION_MODEL_ID, api_key=ROBOFLOW_API_KEY_FIELD)

# --- MSA Detection Model ---
# (For demonstration, we use your workflow info; adjust if needed)
MSA_DETECTION_MODEL = get_model(model_id="football-field-detection-f07vi/14", api_key=ROBOFLOW_API_KEY_FIELD)
# (In a real scenario you might need a different call if the workflow is not a “model”)

from configs.soccer import SoccerPitchConfiguration
from common.view import ViewTransformer
from annotators.soccer import draw_pitch, draw_points_on_pitch, draw_pitch_voronoi_diagram
from common.team import TeamClassifier

try:
    import supervision as sv
except ImportError:
    raise ImportError("Please install the supervision package (pip install supervision)")

# Setup annotators
box_annotator = sv.BoxAnnotator(color=sv.ColorPalette.from_hex(['#FF8C00','#00BFFF','#FF1493','#FFD700']), thickness=2)
label_annotator = sv.LabelAnnotator(color=sv.ColorPalette.from_hex(['#FF8C00','#00BFFF','#FF1493','#FFD700']),
                                    text_color=sv.Color.from_hex('#000000'))

# --- Global Variables ---
stop_rtmp_flag = False
rtmp_thread = None
latest_frame = None

model_active = False
active_model = None
annotated_frame = None

yolo_model = None
yolo_thread = None
rps_thread = None
football_thread = None
football_model = None

recording = False
video_writer = None

BATCH_SIZE = 32
EMBEDDINGS_PROCESSOR = AutoProcessor.from_pretrained('google/siglip-base-patch16-224')
REDUCER = umap.UMAP(n_components=3)
CLUSTERING_MODEL = KMeans(n_clusters=2)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBEDDINGS_MODEL = SiglipVisionModel.from_pretrained('google/siglip-base-patch16-224').to(DEVICE)
team_classifier = TeamClassifier(device=DEVICE)

# For tracking (using ByteTrack)
BALL_ID = 0
tracker = sv.ByteTrack()
tracker.reset()

ellipse_annotator = sv.EllipseAnnotator(color=sv.ColorPalette.from_hex(['#00BFFF','#FF1493','#FFD700']), thickness=2)
label_annotator = sv.LabelAnnotator(color=sv.ColorPalette.from_hex(['#00BFFF','#FF1493','#FFD700']),
                                    text_color=sv.Color.from_hex('#000000'),
                                    text_position=sv.Position.BOTTOM_CENTER)
triangle_annotator = sv.TriangleAnnotator(color=sv.Color.from_hex('#FFD700'), base=25, height=21, outline_thickness=1)

def check_device():
    if torch.cuda.is_available():
        print("Running on GPU")
        return "cuda"
    else:
        print("Running on CPU")
        return "cpu"

DEVICE = check_device()

# --- Speed Tracking Globals ---
# These dictionaries store last position, timestamp and speed stats per player id.
player_last_pos = {}
player_last_time = {}
player_speed_sum = {}
player_speed_count = {}

# --- Logging Functionality ---
log_messages = []
def log(msg):
    timestamp = time.strftime("%H:%M:%S")
    full_msg = f"[{timestamp}] {msg}\n"
    log_messages.append(full_msg)
    # Append to terminal output widget (if exists)
    if terminal_text:
        terminal_text.insert(tk.END, full_msg)
        terminal_text.see(tk.END)
    print(full_msg.strip())

# --- Inference Helper Functions (unchanged) ---
def my_sink(result, video_frame):
    for box in result.get("boxes", []):
        x1, y1, x2, y2 = box
        cv2.rectangle(video_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(video_frame, result.get("prediction", ""),
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    log("RPS Prediction: " + str(result.get("prediction", "")))

def rps_inference():
    global latest_frame, annotated_frame, model_active
    while model_active:
        if latest_frame is not None:
            frame = latest_frame.copy()
            try:
                result = {"boxes": [(50, 50, 150, 150)], "prediction": "rock"}
                my_sink(result, frame)
                annotated_frame = frame
            except Exception as e:
                log(f"RPS inference error: {e}")
        time.sleep(0.03)

def yolo_inference():
    global latest_frame, annotated_frame, model_active, yolo_model
    while model_active:
        if latest_frame is not None:
            frame = latest_frame.copy()
            try:
                results = yolo_model.predict(frame, stream=False)
                if results:
                    result = results[0]
                    annotated = result.plot()
                    annotated_frame = annotated
            except Exception as e:
                log(f"YOLO inference error: {e}")
        time.sleep(0.03)

def football_inference():
    global latest_frame, annotated_frame, model_active, football_model, tracker
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
                    detections.xyxy = detections.xyxy / scale_factor
                    ball_detections = detections[detections.class_id == BALL_ID]
                    ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)
                    all_detections = detections[detections.class_id != BALL_ID]
                    all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
                    all_detections.class_id -= 1
                    tracked_detections = tracker.update_with_detections(detections=all_detections)
                    labels = [f"#{tid}" for tid in tracked_detections.tracker_id]
                    annotated = ellipse_annotator.annotate(scene=frame.copy(), detections=tracked_detections)
                    annotated = label_annotator.annotate(scene=annotated, detections=tracked_detections, labels=labels)
                    annotated = triangle_annotator.annotate(scene=annotated, detections=ball_detections)
                    annotated_frame = annotated
                except Exception as e:
                    log(f"Football inference error: {e}")
        time.sleep(0.03)

def custom_inference():
    global latest_frame, annotated_frame, model_active
    while model_active:
        if latest_frame is not None:
            frame = latest_frame.copy()
            cv2.rectangle(frame, (60, 60), (200, 200), (0, 255, 255), 3)
            cv2.putText(frame, "Custom Workflow", (60, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            annotated_frame = frame
        time.sleep(0.03)

def msa_inference():
    global latest_frame, annotated_frame, model_active, MSA_DETECTION_MODEL
    while model_active:
        if latest_frame is not None:
            frame = latest_frame.copy()
            try:
                result = MSA_DETECTION_MODEL.infer(frame, confidence=0.3)[0]
                detections = sv.Detections.from_inference(result)
                annotated = box_annotator.annotate(scene=frame.copy(), detections=detections)
                annotated_frame = annotated
            except Exception as e:
                log(f"MSA detection inference error: {e}")
        time.sleep(0.03)

# --- Behavioral Map & Speed Tracking ---
def update_behavioral_map():
    global player_last_pos, player_last_time, player_speed_sum, player_speed_count
    if latest_frame is None:
        return

    try:
        result = PLAYER_DETECTION_MODEL.infer(latest_frame, confidence=0.3)[0]
        detections = sv.Detections.from_inference(result)
        # Filter detections for players (assuming class_id==2)
        players_detections = detections[detections.class_id == 2]
        # Update tracker (this returns a Detections object with bounding boxes and tracker ids)
        tracked = tracker.update_with_detections(detections=players_detections)
        # Compute centers manually from tracked.xyxy
        if tracked.xyxy.size > 0:
            centers = np.column_stack(((tracked.xyxy[:, 0] + tracked.xyxy[:, 2]) / 2,
                                        (tracked.xyxy[:, 1] + tracked.xyxy[:, 3]) / 2))
        else:
            centers = np.array([])
        ids = tracked.tracker_id
    except Exception as e:
        log(f"Behavioral map detection error: {e}")
        return

    current_time = time.time()
    behav_canvas.delete("all")
    canvas_width = int(behav_canvas['width'])
    canvas_height = int(behav_canvas['height'])

    for idx, pid in enumerate(ids):
        pos = centers[idx]
        canvas_x = int(pos[0] * canvas_width / latest_frame.shape[1])
        canvas_y = int(pos[1] * canvas_height / latest_frame.shape[0])

        if pid in player_last_pos:
            dt = current_time - player_last_time[pid]
            if dt > 0:
                dist = np.linalg.norm(np.array(pos) - np.array(player_last_pos[pid]))
                speed = dist / dt
                player_speed_sum[pid] = player_speed_sum.get(pid, 0) + speed
                player_speed_count[pid] = player_speed_count.get(pid, 0) + 1
            else:
                speed = 0
        else:
            speed = 0
        player_last_pos[pid] = pos
        player_last_time[pid] = current_time
        avg_speed = player_speed_sum.get(pid, 0) / player_speed_count.get(pid, 1)

        team_color = "blue" if canvas_x < canvas_width / 2 else "red"
        r = 10
        behav_canvas.create_oval(canvas_x - r, canvas_y - r, canvas_x + r, canvas_y + r,
                                   fill=team_color, outline="")
        behav_canvas.create_text(canvas_x, canvas_y - 15, text=f"ID:{pid} {avg_speed:.1f}", fill="white", font=("Helvetica", 8))

    stats = "Speed Stats: " + ", ".join([f"ID {pid}: {player_speed_sum.get(pid, 0)/player_speed_count.get(pid, 1):.1f}" for pid in ids])
    log(stats)

# --- Update Field Map (Behavioral Map) Loop ---
def behavioral_map_loop():
    # Only update if the behavioral map is active
    while behav_map_active:
        update_behavioral_map()
        time.sleep(0.5)

# --- Video/Stream Handling ---
def load_video_files():
    # List files in "videos" folder
    if os.path.isdir("videos"):
        files = [f for f in os.listdir("videos") if f.lower().endswith(('.mp4','.avi','.mov'))]
        return files
    return []

def show_stream():
    global stop_rtmp_flag, rtmp_thread, latest_frame
    rtmp_url = rtsp_entry.get().strip()
    if not rtmp_url:
        # If no RTMP link entered, try file selection from combobox
        selected = video_combo.get().strip()
        if selected:
            rtmp_url = os.path.join("videos", selected)
        else:
            messagebox.showwarning("No Video", "Please enter an RTMP link or select a video file.")
            return
    stop_rtmp()
    stop_rtmp_flag = False

    # Update right panel to show video stream
    def rtmp_loop():
        cap = cv2.VideoCapture(rtmp_url)
        if not cap.isOpened():
            log("Could not open video stream.")
            return
        while not stop_rtmp_flag:
            ret, frame = cap.read()
            if not ret:
                log("Failed to read frame from video stream.")
                break
            global latest_frame
            latest_frame = frame
            time.sleep(0.03)
        cap.release()
        log("Video stream ended.")
    rtmp_thread = threading.Thread(target=rtmp_loop, daemon=True)
    rtmp_thread.start()

def stop_rtmp():
    global stop_rtmp_flag, rtmp_thread, latest_frame
    stop_rtmp_flag = True
    if rtmp_thread and rtmp_thread.is_alive():
        rtmp_thread.join()
    latest_frame = None

# --- Model Handling (unchanged branches, with added MSA Detection) ---
def on_run_model():
    global model_active, active_model, yolo_model, yolo_thread, rps_thread, football_thread, football_model
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
        log("YOLO Pose inference started.")
    elif chosen_model == "football-players-detection-3zvbc/11":
        if football_model is None:
            try:
                football_model = get_model(model_id=PLAYER_DETECTION_MODEL_ID, api_key=ROBOFLOW_API_KEY)
            except Exception as e:
                messagebox.showerror("Model Error", f"Failed to load football model: {e}")
                model_active = False
                return
        football_thread = threading.Thread(target=football_inference, daemon=True)
        football_thread.start()
        log("Football inference started.")
    elif chosen_model == "Custom Workflow":
        try:
            custom_model = get_model(workspace_name="noveternum", workflow_id="custom-workflow")
        except Exception as e:
            messagebox.showerror("Model Error", f"Failed to load Custom Workflow model: {e}")
            model_active = False
            return
        threading.Thread(target=custom_inference, daemon=True).start()
        log("Custom Workflow inference started.")
    elif chosen_model == "MSA Detection":
        try:
            _ = get_model(workspace_name="noveternum", workflow_id="custom-workflow")
        except Exception as e:
            messagebox.showerror("Model Error", f"Failed to load MSA Detection model: {e}")
            model_active = False
            return
        threading.Thread(target=msa_inference, daemon=True).start()
        log("MSA Detection inference started.")
    else:
        log(f"Model overlay activated: {chosen_model}")

def stop_model():
    global model_active, active_model, yolo_thread, rps_thread, football_thread, annotated_frame
    model_active = False
    active_model = None
    if yolo_thread and yolo_thread.is_alive():
        yolo_thread.join()
    if rps_thread and rps_thread.is_alive():
        rps_thread.join()
    if football_thread and football_thread.is_alive():
        football_thread.join()
    annotated_frame = None
    log("Model stopped.")

def on_start_recording():
    global recording, video_writer, latest_frame
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
    log("Recording started.")

def on_stop_recording():
    global recording, video_writer
    if recording:
        recording = False
        if video_writer:
            video_writer.release()
            video_writer = None
        log("Recording stopped.")
    else:
        messagebox.showinfo("Recording", "Recording is not in progress.")

# --- GUI Layout Setup ---
root = tk.Tk()
root.title("Full-Screen Application")

# Make window full-screen
root.state("zoomed")
root.configure(bg="black")

# Main layout: Left panel, Right panel, and Bottom terminal
left_panel = tk.Frame(root, bg="black", width=300)
left_panel.pack(side=tk.LEFT, fill=tk.Y)
right_panel = tk.Frame(root, bg="gray20")
right_panel.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
bottom_terminal = tk.Frame(root, bg="black", height=150)
bottom_terminal.pack(side=tk.BOTTOM, fill=tk.X)

# Left panel is split into Control Panel (top) and Behavioral Map (bottom)
control_panel = tk.Frame(left_panel, bg="black")
control_panel.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
# Behavioral Map canvas (initially inactive: gray background)
behav_canvas = tk.Canvas(left_panel, bg="gray30", width=280, height=200)
behav_canvas.pack(side=tk.BOTTOM, padx=10, pady=10)
behav_map_active = False  # Toggle state

def toggle_behavioral_map():
    global behav_map_active
    behav_map_active = not behav_map_active
    if behav_map_active:
        behav_canvas.config(bg="black")
        log("Behavioral Map activated.")
        # Start behavioral map update thread if not already running
        threading.Thread(target=behavioral_map_loop, daemon=True).start()
    else:
        behav_canvas.config(bg="gray30")
        log("Behavioral Map deactivated.")

btn_toggle_behav = ttk.Button(control_panel, text="Toggle Behavioral Map", command=toggle_behavioral_map)
btn_toggle_behav.pack(pady=(0,10))

# Video input controls in control_panel
rtsp_label = ttk.Label(control_panel, text="RTMP Link:")
rtsp_label.pack(anchor="w")
rtsp_entry = ttk.Entry(control_panel, width=40)
rtsp_entry.pack(pady=(0,5))
# Also add a combobox for local videos
ttk.Label(control_panel, text="Or choose a local video:").pack(anchor="w")
video_files = load_video_files()
video_combo = ttk.Combobox(control_panel, values=video_files, state="readonly", width=37)
video_combo.pack(pady=(0,10))

# FPS slider
ttk.Label(control_panel, text="FPS:").pack(anchor="w")
fps_value = tk.IntVar(value=60)
fps_slider = tk.Scale(control_panel, from_=1, to=100, orient=tk.HORIZONTAL, variable=fps_value,
                      bg="black", fg="white", highlightbackground="black")
fps_slider.pack(fill=tk.X, pady=(0,10))

# Model selection
ttk.Label(control_panel, text="Choose Model:").pack(anchor="w")
model_options = ["YOLO Pose", "football-players-detection-3zvbc/11", "Custom Workflow", "MSA Detection"]
model_var = tk.StringVar(value=model_options[0])
model_combo = ttk.Combobox(control_panel, textvariable=model_var, values=model_options, state="readonly", width=37)
model_combo.pack(pady=(0,10))

# Model control buttons
model_btn_frame = tk.Frame(control_panel, bg="black")
model_btn_frame.pack(pady=5)
btn_run_model = ttk.Button(model_btn_frame, text="Start Model", command=on_run_model)
btn_run_model.grid(row=0, column=0, padx=5)
btn_stop_model = ttk.Button(model_btn_frame, text="Stop Model", command=stop_model)
btn_stop_model.grid(row=0, column=1, padx=5)

# Recording buttons
record_btn_frame = tk.Frame(control_panel, bg="black")
record_btn_frame.pack(pady=5)
btn_start_recording = ttk.Button(record_btn_frame, text="Start Recording", command=on_start_recording)
btn_start_recording.grid(row=0, column=0, padx=5)
btn_stop_recording = ttk.Button(record_btn_frame, text="Stop Recording", command=on_stop_recording)
btn_stop_recording.grid(row=0, column=1, padx=5)

# Stream control buttons
stream_btn_frame = tk.Frame(control_panel, bg="black")
stream_btn_frame.pack(pady=5)
btn_show_stream = ttk.Button(stream_btn_frame, text="Show Stream", command=show_stream)
btn_show_stream.grid(row=0, column=0, padx=5)
btn_stop_stream = ttk.Button(stream_btn_frame, text="Stop Stream", command=stop_rtmp)
btn_stop_stream.grid(row=0, column=1, padx=5)

# --- Right Panel: Video Stream Display ---
video_label = tk.Label(right_panel, text="No Stream Currently", bg="gray20", fg="white")
video_label.pack(expand=True, fill=tk.BOTH)

# --- Bottom Terminal Panel ---
terminal_text = tk.Text(bottom_terminal, bg="black", fg="lime", height=8)
terminal_text.pack(expand=True, fill=tk.BOTH)
terminal_text.insert(tk.END, "Terminal output...\n")
terminal_text.config(state=tk.DISABLED)

# Override print to also log in terminal (optional)
# You can also use the log() function to add messages.

# --- Update Frame Function ---
def update_frame():
    if right_panel.winfo_exists():
        if latest_frame is None:
            video_label.config(text="No Stream Currently", image="", bg="gray20")
        else:
            try:
                # If model is active and we have annotated_frame, use it; otherwise use latest_frame
                frame_to_show = annotated_frame.copy() if (model_active and annotated_frame is not None) else latest_frame.copy()
                # Optionally overlay model name
                if model_active:
                    cv2.putText(frame_to_show, f"Model: {active_model}",
                                (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
                frame_rgb = cv2.cvtColor(frame_to_show, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                video_label.config(image=imgtk, text="")
                video_label.image = imgtk
                # Optionally, write frame to video_writer if recording
                if recording and video_writer is not None:
                    video_writer.write(frame_to_show)
            except Exception as e:
                log(f"Display frame error: {e}")
    delay = int(1000 / fps_value.get())
    root.after(delay, update_frame)

update_frame()
root.bind("<Escape>", lambda event: root.quit())
root.protocol("WM_DELETE_WINDOW", lambda: [stop_rtmp(), root.destroy()])
root.mainloop()
