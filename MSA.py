import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import time
import threading
from ultralytics import YOLO  # Make sure to install ultralytics (pip install ultralytics)

#############################
# Global Variables
#############################
stop_rtmp_flag = False
rtmp_thread = None

latest_frame = None  # The most recent frame from the RTMP stream

# Global variables for model overlay/inference
model_active = False       # Whether a model overlay/inference is active
active_model = None        # The name of the active model (string)
annotated_frame = None     # The frame with YOLO annotations (if applicable)
yolo_model = None          # The YOLO model instance
yolo_thread = None         # Thread for running YOLO inference

# Global variables for recording
recording = False
video_writer = None

#############################
# Helper functions for button commands
#############################
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
    
    # Check if a frame is available to record
    if latest_frame is None:
        messagebox.showwarning("No Video", "No video frame available to record yet.")
        return

    if recording:
        messagebox.showinfo("Recording", "Recording is already in progress.")
        return

    # Get frame dimensions from the latest frame
    height, width, _ = latest_frame.shape
    # Define the codec and create VideoWriter object.
    # Adjust fps (here set to 10) and filename as needed.
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
    """
    Bind ESC to quit the app (or exit fullscreen).
    """
    root.quit()

#############################
# Main Functions
#############################
def show_stream():
    """
    Called when 'Show Stream' is clicked.
    Starts an RTMP thread that continuously updates latest_frame.
    """
    global stop_rtmp_flag, rtmp_thread

    print("[INFO] 'Show Stream' button clicked.")

    rtmp_url = url_entry.get()
    if not rtmp_url:
        messagebox.showwarning("No RTMP URL", "Please enter an RTMP URL.")
        return

    # Stop any running RTMP capture.
    stop_rtmp()

    stop_rtmp_flag = False

    def rtmp_loop():
        print(f"[INFO] RTMP loop started with URL = {rtmp_url}")
        cap = cv2.VideoCapture(rtmp_url)
        if not cap.isOpened():
            print("[ERROR] Could not open RTMP stream.")
            return

        while not stop_rtmp_flag:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read frame from RTMP stream.")
                break
            global latest_frame
            latest_frame = frame
            time.sleep(0.03)  # Approximately 30 fps

        cap.release()
        print("[INFO] RTMP capture loop ended.")

    rtmp_thread = threading.Thread(target=rtmp_loop, daemon=True)
    rtmp_thread.start()
    print("[INFO] RTMP thread started.")

def stop_rtmp():
    """
    Stops the RTMP thread and clears the displayed frame.
    """
    global stop_rtmp_flag, rtmp_thread, latest_frame
    print("[INFO] stop_rtmp() called.")

    stop_rtmp_flag = True
    if rtmp_thread and rtmp_thread.is_alive():
        rtmp_thread.join()
    rtmp_thread = None

    latest_frame = None
    video_label.config(image="", text="No Video", fg="white")
    print("[INFO] RTMP capture stopped (thread joined).")

def run_model():
    """
    Called when 'Start Model' is clicked.
    For "YOLO Pose", the YOLO model is loaded and inference starts in a dedicated thread.
    For other models, a simple text overlay is drawn.
    """
    global model_active, active_model, yolo_model, yolo_thread
    print("[INFO] Start Model clicked.")
    if latest_frame is None:
        messagebox.showwarning("No Stream", "RTMP stream is not active. Please start the stream first.")
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
        # Start the YOLO inference thread
        yolo_thread = threading.Thread(target=yolo_inference, daemon=True)
        yolo_thread.start()
        print("[INFO] YOLO Pose inference started.")
    else:
        # For other models, we simply overlay the model name.
        print(f"[INFO] Model overlay activated: {chosen_model}")

def stop_model():
    """
    Stops any active model overlay/inference.
    If YOLO inference is running, signal it to stop.
    """
    global model_active, active_model, yolo_thread
    print("[INFO] Stop Model clicked.")
    model_active = False
    active_model = None
    # If the YOLO thread is running, join it.
    if yolo_thread and yolo_thread.is_alive():
        yolo_thread.join()
    # Clear the annotated frame
    global annotated_frame
    annotated_frame = None

#############################
# YOLO Inference Thread Function
#############################
def yolo_inference():
    """
    Runs YOLO Pose inference on the latest_frame in a loop.
    Each frame is processed and the annotated result is stored in annotated_frame.
    """
    global latest_frame, annotated_frame, model_active, yolo_model
    while model_active:
        if latest_frame is not None:
            # Copy the current frame for inference
            frame = latest_frame.copy()
            try:
                # Predict on the single frame; stream=False returns a list of results.
                results = yolo_model.predict(frame, stream=False)
                if results:
                    # Assume one result and get the annotated frame.
                    result = results[0]
                    annotated = result.plot()  # This returns an annotated numpy array.
                    annotated_frame = annotated
            except Exception as e:
                print(f"[ERROR] YOLO inference error: {e}")
        time.sleep(0.03)

#############################
# Update Frame (Display Video)
#############################
def update_frame():
    global latest_frame, video_writer, recording, model_active, active_model, annotated_frame
    if latest_frame is None:
        video_label.config(text="No Video", image="")
    else:
        try:
            # If YOLO Pose is active and we have an annotated frame, display it.
            if model_active and active_model == "YOLO Pose" and annotated_frame is not None:
                frame_to_show = annotated_frame.copy()
            else:
                # Otherwise, use the latest RTMP frame.
                frame_to_show = latest_frame.copy()
                # For other models, overlay the model name.
                if model_active and active_model != "YOLO Pose":
                    cv2.putText(
                        frame_to_show,
                        f"Model: {active_model}",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
            frame_rgb = cv2.cvtColor(frame_to_show, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            video_label.config(image=imgtk, text="")
            video_label.image = imgtk  # Keep a reference!
            
            # If recording is active, write the combined frame (stream + overlay) to file.
            if recording and video_writer is not None:
                video_writer.write(frame_to_show)
        except Exception as e:
            print("[ERROR] Could not display frame:", e)
            video_label.config(image="", text="No Video")
    root.after(50, update_frame)

#############################
# Build the GUI
#############################
root = tk.Tk()
root.title("Full-Screen Dark UI")

# Make it full-screen.
root.attributes("-fullscreen", True)

# Dark theme styling with Ttk.
style = ttk.Style()
style.theme_use("clam")
root.configure(bg="black")

style.configure("TFrame", background="black")
style.configure(
    "TButton",
    background="gray25",
    foreground="white",
    bordercolor="gray25",
    focusthickness=3,
    focuscolor="gray25",
)
style.configure("TLabel", background="black", foreground="white")
style.configure("TCombobox", fieldbackground="gray25", foreground="white")
style.configure("TEntry", fieldbackground="gray25", foreground="white")
style.map(
    "TButton",
    foreground=[("active", "white"), ("disabled", "grey")],
    background=[("active", "gray40"), ("disabled", "gray20")],
)

root.bind("<Escape>", on_escape)

# Left/Right frames.
left_frame = ttk.Frame(root, width=300)
left_frame.pack(side=tk.LEFT, fill=tk.Y)

right_frame = ttk.Frame(root)
right_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

# Top/bottom split in right frame.
top_frame = ttk.Frame(right_frame)
top_frame.pack(side=tk.TOP, expand=True, fill=tk.BOTH)

bottom_frame = ttk.Frame(right_frame, height=200)
bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)

# Left Menu Panel.
url_label = ttk.Label(left_frame, text="RTMP URL:")
url_label.pack(pady=(20, 5))

url_entry = ttk.Entry(left_frame, width=25)
url_entry.insert(0, "rtmp://127.0.0.1:1935/stream")
url_entry.pack(pady=(0, 20))

stream_button_frame = ttk.Frame(left_frame)
stream_button_frame.pack(pady=10)

btn_show_stream = ttk.Button(stream_button_frame, text="Show Stream", command=on_show_stream)
btn_show_stream.grid(row=0, column=0, padx=5)

btn_stop_stream = ttk.Button(stream_button_frame, text="Stop Stream", command=on_stop_stream)
btn_stop_stream.grid(row=0, column=1, padx=5)

model_label = ttk.Label(left_frame, text="Choose Model:")
model_label.pack(pady=(20, 5))

# Add "YOLO Pose" as an option along with your other models.
model_options = [
    "YOLO Pose",
    "rock-paper-scissors-sxsw/14",
    "football-players-detection-3zvbc/12",
    "some-other-model-id/1"
]
model_var = tk.StringVar()
model_combo = ttk.Combobox(left_frame, textvariable=model_var, values=model_options, state="readonly")
model_combo.pack(pady=(0, 20))
model_combo.current(0)  # Optionally set a default (here "YOLO Pose")

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

# Right Side: Video display up top.
video_label = tk.Label(top_frame, text="No Video", bg="black", fg="white")
video_label.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

# Placeholder at bottom.
bottom_label = ttk.Label(bottom_frame, text="Placeholder for future content.")
bottom_label.pack(padx=10, pady=10)

# Kick off frame updates.
update_frame()

# Start the Tkinter main loop.
root.mainloop()
