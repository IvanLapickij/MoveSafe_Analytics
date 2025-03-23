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
from datetime import datetime
# For Roboflow football model inference (using inference-gpu)
from inference import get_model

ROBOFLOW_API_KEY = "tvZVhjN9hMWkURbVo84w"
PLAYER_DETECTION_MODEL_ID = "movesafep4/3"  # Use your model ID
# Load the player detection model
PLAYER_DETECTION_MODEL = get_model(model_id=PLAYER_DETECTION_MODEL_ID, api_key=ROBOFLOW_API_KEY)

# Initialize field detection model
ROBOFLOW_API_KEY_FIELD = ROBOFLOW_API_KEY
FIELD_DETECTION_MODEL_ID = "football-field-detection-f07vi/14"
FIELD_DETECTION_MODEL = get_model(model_id=FIELD_DETECTION_MODEL_ID, api_key=ROBOFLOW_API_KEY_FIELD)

# Additional imports for pitch keypoint detection and projection
from configs.soccer import SoccerPitchConfiguration
from common.view import ViewTransformer
from annotators.soccer import draw_pitch, draw_points_on_pitch
from common.team import TeamClassifier

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

# Clustering globals
BATCH_SIZE = 32
EMBEDDINGS_MODEL = SiglipVisionModel.from_pretrained('google/siglip-base-patch16-224').to('cuda' if torch.cuda.is_available() else 'cpu')
EMBEDDINGS_PROCESSOR = AutoProcessor.from_pretrained('google/siglip-base-patch16-224')
REDUCER = umap.UMAP(n_components=3)
CLUSTERING_MODEL = KMeans(n_clusters=2)

# Team classifier (for later use in team projection)
team_classifier = TeamClassifier(device="cuda")

# FPS control (default 60 FPS)
root = tk.Tk()
fps_value = tk.IntVar(root, value=60)

# --- Additional globals for Football Tracking and Annotation ---
BALL_ID = 0  # Assumed ball class ID
tracker = sv.ByteTrack()  # Initialize ByteTrack tracker
tracker.reset()

ellipse_annotator = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
    thickness=2
)
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
    for box in result.get("boxes", []):
        x1, y1, x2, y2 = box
        cv2.rectangle(video_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(video_frame, result.get("prediction", ""),
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    print("RPS Prediction:", result)

def rps_inference():
    global latest_frame, annotated_frame, model_active
    while model_active:
        if latest_frame is not None:
            frame = latest_frame.copy()
            try:
                result = {
                    "boxes": [(50, 50, 150, 150)],
                    "prediction": "rock"
                }
                my_sink(result, frame)
                annotated_frame = frame
            except Exception as e:
                print(f"[ERROR] RPS inference error: {e}")
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
                print(f"[ERROR] YOLO inference error: {e}")
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
                    labels = [f"#{tracker_id}" for tracker_id in tracked_detections.tracker_id]
                    annotated = ellipse_annotator.annotate(scene=frame.copy(), detections=tracked_detections)
                    annotated = label_annotator.annotate(scene=annotated, detections=tracked_detections, labels=labels)
                    annotated = triangle_annotator.annotate(scene=annotated, detections=ball_detections)
                    annotated_frame = annotated
                except Exception as e:
                    print(f"[ERROR] Football inference error: {e}")
        time.sleep(0.03)

# --- New Function: Split Teams via Clustering ---
def split_teams():
    global football_model
    if latest_frame is None:
        messagebox.showwarning("No Video", "Video stream is not active.")
        return
    if football_model is None:
        try:
            football_model = get_model(model_id=PLAYER_DETECTION_MODEL_ID, api_key=ROBOFLOW_API_KEY)
        except Exception as e:
            messagebox.showerror("Model Error", f"Failed to load football model: {e}")
            return

    messagebox.showinfo("Info", "Capturing 5 seconds of footage for team split...")
    crops = []
    start_time = time.time()
    captured_frames = []
    while time.time() - start_time < 5:
        if latest_frame is not None:
            captured_frames.append(latest_frame.copy())
            time.sleep(1)
    for frame in captured_frames:
        try:
            result = football_model.infer(frame, confidence=0.3)[0]
            detections = sv.Detections.from_inference(result)
            players_detections = detections[detections.class_id != BALL_ID]
            players_detections = players_detections.with_nms(threshold=0.5, class_agnostic=True)
            for xyxy in players_detections.xyxy:
                crop = sv.crop_image(frame, xyxy)
                crops.append(crop)
        except Exception as e:
            print(f"[ERROR] Player detection error: {e}")
    if not crops:
        messagebox.showwarning("No Players", "No players detected in the captured footage.")
        return

    pil_crops = [sv.cv2_to_pillow(crop) for crop in crops]
    embeddings_list = []
    for batch in chunked(pil_crops, BATCH_SIZE):
        try:
            inputs = EMBEDDINGS_PROCESSOR(images=batch, return_tensors="pt").to(EMBEDDINGS_MODEL.device)
            with torch.no_grad():
                outputs = EMBEDDINGS_MODEL(**inputs)
            embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
            embeddings_list.append(embeddings)
        except Exception as e:
            print(f"[ERROR] Embedding extraction error: {e}")
    data = np.concatenate(embeddings_list)
    projections = REDUCER.fit_transform(data)
    clusters = CLUSTERING_MODEL.fit_predict(projections)
    grid_img = sv.plot_images_grid(pil_crops[:100], grid_size=(10, 10))
    save_path = os.path.join(os.getcwd(), "teams_split.png")
    grid_img.savefig(save_path)
    messagebox.showinfo("Teams Split", f"Team split complete. Grid image saved at:\n{save_path}")

def on_split_teams():
    threading.Thread(target=split_teams, daemon=True).start()

# --- New Function: Pitch Keypoint Detection and Projection ---
def pitch_keypoint_detection():
    global latest_frame, FIELD_DETECTION_MODEL
    if latest_frame is None:
        messagebox.showwarning("No Video", "Video stream is not active.")
        return
    frame = latest_frame.copy()
    try:
        result = FIELD_DETECTION_MODEL.infer(frame, confidence=0.3)[0]
    except Exception as e:
        messagebox.showerror("Field Model Error", f"Error during field detection: {e}")
        return
    key_points = sv.KeyPoints.from_inference(result)
    confidence_filter = key_points.confidence[0] > 0.5
    if not np.any(confidence_filter):
        messagebox.showwarning("Low Confidence", "No high confidence keypoints detected.")
        return
    frame_reference_points = key_points.xy[0][confidence_filter]
    frame_reference_key_points = sv.KeyPoints(xy=frame_reference_points[np.newaxis, ...])
    vertex_annotator_pitch = sv.VertexAnnotator(color=sv.Color.from_hex('#FF1493'), radius=8)
    annotated_frame = frame.copy()
    annotated_frame = vertex_annotator_pitch.annotate(scene=annotated_frame, key_points=frame_reference_key_points)
    save_path = os.path.join(os.getcwd(), "pitch_keypoints.png")
    cv2.imwrite(save_path, annotated_frame)
    messagebox.showinfo("Pitch Keypoints", f"Pitch keypoints saved at:\n{save_path}")
    
    CONFIG = SoccerPitchConfiguration()
    pitch_vertices = np.array(CONFIG.vertices)
    pitch_reference_points = pitch_vertices[confidence_filter]
    transformer = ViewTransformer(source=frame_reference_points, target=pitch_reference_points)
    pitch_all_points = pitch_vertices
    frame_all_points = transformer.transform_points(points=pitch_all_points)
    frame_all_key_points = sv.KeyPoints(xy=frame_all_points[np.newaxis, ...])
    edge_annotator_pitch = sv.EdgeAnnotator(color=sv.Color.from_hex('#00BFFF'), thickness=2, edges=CONFIG.edges)
    annotated_frame = frame.copy()
    annotated_frame = edge_annotator_pitch.annotate(scene=annotated_frame, key_points=frame_all_key_points)
    save_path2 = os.path.join(os.getcwd(), "projected_pitch_lines.png")
    cv2.imwrite(save_path2, annotated_frame)
    messagebox.showinfo("Projected Pitch Lines", f"Projected pitch lines saved at:\n{save_path2}")

def on_pitch_detection():
    threading.Thread(target=pitch_keypoint_detection, daemon=True).start()

# --- New Globals and Functions for Live Field Map ---
field_map_window = None
field_map_label = None
field_map_active = False
field_transformer = None
CONFIG = SoccerPitchConfiguration()

def compute_field_transformer(frame):
    try:
        result = FIELD_DETECTION_MODEL.infer(frame, confidence=0.3)[0]
        key_points = sv.KeyPoints.from_inference(result)
        conf_filter = key_points.confidence[0] > 0.5
        if not np.any(conf_filter):
            return None
        frame_ref_points = key_points.xy[0][conf_filter]
        pitch_ref_points = np.array(CONFIG.vertices)[conf_filter]
        transformer = ViewTransformer(source=frame_ref_points, target=pitch_ref_points)
        return transformer
    except Exception as e:
        print(f"[ERROR] Computing field transformer: {e}")
        return None

def update_field_map():
    global field_map_window, field_map_label, field_map_active, field_transformer
    while field_map_active:
        if latest_frame is None:
            time.sleep(0.1)
            continue
        frame = latest_frame.copy()
        if field_transformer is None:
            field_transformer = compute_field_transformer(frame)
            if field_transformer is None:
                time.sleep(1)
                continue
        try:
            result = PLAYER_DETECTION_MODEL.infer(frame, confidence=0.3)[0]
            detections = sv.Detections.from_inference(result)
            players_detections = detections[detections.class_id == 2]  # PLAYER_ID assumed as 2
            players_xy = players_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            pitch_players_xy = field_transformer.transform_points(points=players_xy)
        except Exception as e:
            print(f"[ERROR] Live field map detection: {e}")
            pitch_players_xy = np.empty((0,2))
        try:
            pitch_img = draw_pitch(CONFIG, background_color=sv.Color.WHITE, line_color=sv.Color.BLACK)
            pitch_img = draw_points_on_pitch(
                config=CONFIG,
                xy=pitch_players_xy,
                face_color=sv.Color.from_hex('00BFFF'),
                edge_color=sv.Color.BLACK,
                radius=16,
                pitch=pitch_img
            )
            field_map_pil = Image.fromarray(pitch_img)
            imgtk = ImageTk.PhotoImage(image=field_map_pil)
            field_map_label.config(image=imgtk)
            field_map_label.image = imgtk
        except Exception as e:
            print(f"[ERROR] Drawing field map: {e}")
        time.sleep(1)

def on_display_field():
    global field_map_window, field_map_label, field_map_active, field_transformer
    if field_map_window is None or not field_map_window.winfo_exists():
        field_map_window = tk.Toplevel(root)
        field_map_window.title("Live Field Map")
        field_map_label = tk.Label(field_map_window, text="Initializing...", bg="white")
        field_map_label.pack(expand=True, fill=tk.BOTH)
        field_map_window.geometry("600x400")
        field_transformer = None
    field_map_active = True
    threading.Thread(target=update_field_map, daemon=True).start()

def stop_field_map():
    global field_map_active
    field_map_active = False

def on_field_map_close():
    stop_field_map()
    if field_map_window is not None:
        field_map_window.destroy()

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

# --- Main Functions for Stream and Model Handling ---
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
        print("[INFO] Football inference started.")
    else:
        print(f"[INFO] Model overlay activated: {chosen_model}")

def stop_model():
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

# --- Helper function to get video files ---
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

# Create left control panel frame
left_frame = ttk.Frame(root, width=300)
left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

# --- Video Input Section (Single Editable Combobox) ---
input_frame = ttk.Frame(left_frame, padding=10)
input_frame.pack(fill=tk.X)

# Label for the input
url_label = ttk.Label(input_frame, text="VIDEO URL or File:")
url_label.grid(row=0, column=0, sticky=tk.W, pady=(0,5))

# Editable combobox for video input.
url_combo = ttk.Combobox(input_frame, state="normal", width=40)
url_combo.grid(row=1, column=0, padx=(0,5), pady=(0,10))

def get_video_files():
    video_folder = "videos"
    if not os.path.exists(video_folder):
        return []
    # Filter for common video file extensions
    video_files = [os.path.join("videos", f) for f in os.listdir(video_folder) 
                   if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    return video_files

def update_url_combo():
    video_files = get_video_files()
    # Set the combobox values to the list of local video files.
    url_combo['values'] = video_files
    # If there are local videos, set the default value.
    if video_files:
        url_combo.set(video_files[0])
    else:
        url_combo.set("")

# Initially populate the combobox with local video files.
update_url_combo()

btn_show_stream = ttk.Button(input_frame, text="Show Stream", command=on_show_stream)
btn_show_stream.grid(row=2, column=0, columnspan=2, pady=(10,0))

# Other controls in left panel
fps_label = ttk.Label(left_frame, text="FPS:")
fps_label.pack(pady=(10, 0))
fps_slider = tk.Scale(left_frame, from_=1, to=100, orient=tk.HORIZONTAL, variable=fps_value)
fps_slider.pack(pady=(0, 10))

stream_button_frame = ttk.Frame(left_frame)
stream_button_frame.pack(pady=10)
btn_show_stream_ctrl = ttk.Button(stream_button_frame, text="Show Stream", command=on_show_stream)
btn_show_stream_ctrl.grid(row=0, column=0, padx=5)
btn_stop_stream = ttk.Button(stream_button_frame, text="Stop Stream", command=on_stop_stream)
btn_stop_stream.grid(row=0, column=1, padx=5)

model_label = ttk.Label(left_frame, text="Choose Model:")
model_label.pack(pady=(20, 5))
model_options = [
    "YOLO Pose",
    "football-players-detection-3zvbc/11"
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

extra_button_frame = ttk.Frame(left_frame)
extra_button_frame.pack(pady=10)
btn_split_teams = ttk.Button(extra_button_frame, text="Split Teams", command=on_split_teams)
btn_split_teams.grid(row=0, column=0, padx=5)
btn_pitch_detect = ttk.Button(extra_button_frame, text="Pitch Detection", command=on_pitch_detection)
btn_pitch_detect.grid(row=0, column=1, padx=5)
btn_live_field = ttk.Button(extra_button_frame, text="Display Field", command=on_display_field)
btn_live_field.grid(row=0, column=2, padx=5)

# Right frame for video display
right_frame = ttk.Frame(root)
right_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

top_frame = ttk.Frame(right_frame)
top_frame.pack(side=tk.TOP, expand=True, fill=tk.BOTH)

bottom_frame = ttk.Frame(right_frame, height=200)
bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)

video_label = tk.Label(top_frame, text="No Video", bg="black", fg="white")
video_label.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

bottom_label = ttk.Label(bottom_frame, text="Placeholder for future content.")
bottom_label.pack(padx=10, pady=10)

update_frame()
root.protocol("WM_DELETE_WINDOW", lambda: [stop_field_map(), root.destroy()])
root.mainloop()
