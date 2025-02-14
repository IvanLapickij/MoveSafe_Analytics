import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import time
import threading

#############################
# Global Variables
#############################
stop_rtmp_flag = False
rtmp_thread = None

latest_frame = None

stop_pipeline_flag = False
pipeline_thread = None

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
    Called when 'Show Stream' button is clicked.
    1) Stop any existing RTMP thread or pipeline.
    2) Start a new thread that captures from the RTMP URL.
    """
    global stop_rtmp_flag, rtmp_thread

    print("[INFO] 'Show Stream' button clicked.")

    rtmp_url = url_entry.get()
    if not rtmp_url:
        messagebox.showwarning("No RTMP URL", "Please enter an RTMP URL.")
        return

    # Stop any running pipeline or previous RTMP capture.
    stop_rtmp()
    stop_model()

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
    Signals the RTMP capture thread to stop and waits for it to exit.
    Also clears the video frame in the GUI.
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
    Called when 'Start Model' button is clicked.
    Stops any active RTMP stream and previous pipeline, then starts the model inference pipeline.
    """
    print("[INFO] Start Model clicked.")
    # Stop RTMP so that only the pipeline runs.
    stop_rtmp()
    # Stop any old pipeline if running.
    stop_model()
    # Start the model inference pipeline.
    start_pipeline_inference()


def stop_model():
    """
    Stops the inference pipeline if it is running.
    """
    global stop_pipeline_flag, pipeline_thread, latest_frame
    print("[INFO] stop_model() called.")

    stop_pipeline_flag = True
    if pipeline_thread and pipeline_thread.is_alive():
        pipeline_thread.join()
        pipeline_thread = None
        print("[INFO] Pipeline thread joined successfully.")

    # Clear the frame so the GUI can show "No Video"
    latest_frame = None
    video_label.config(image="", text="No Video", fg="white")


#############################
# Pipeline (Model Inference) Implementation
#############################
def start_pipeline_inference():
    """
    Starts a new thread that simulates running a model inference pipeline.
    In this demo, we capture from the local webcam and overlay the chosen model name.
    """
    global pipeline_thread, stop_pipeline_flag
    print("[INFO] start_pipeline_inference called.")

    # Make sure no previous pipeline is running.
    stop_model()

    stop_pipeline_flag = False

    def pipeline_loop():
        global latest_frame, stop_pipeline_flag
        print("[INFO] Pipeline loop started.")

        # Open a video capture source (e.g., webcam 0)
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("[ERROR] Could not open video source for model inference.")
            stop_pipeline_flag = True
            return

        # Get the selected model from the combobox.
        chosen_model = model_var.get()
        print(f"[INFO] Running inference with model: {chosen_model}")

        while not stop_pipeline_flag:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Could not read frame from video source.")
                break

            # Simulate inference by overlaying text on the frame.
            if chosen_model:
                cv2.putText(
                    frame,
                    f"Model: {chosen_model}",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            latest_frame = frame
            time.sleep(0.03)  # ~30 fps

        cap.release()
        print("[INFO] Pipeline loop ended.")

    pipeline_thread = threading.Thread(target=pipeline_loop, daemon=True)
    pipeline_thread.start()
    print("[INFO] Pipeline thread started.")


#############################
# Update Frame (Display Video)
#############################
def update_frame():
    global latest_frame, video_writer, recording
    if latest_frame is None:
        video_label.config(text="No Video", image="")
    else:
        try:
            # Convert from BGR to RGB for Tkinter/PIL.
            frame_rgb = cv2.cvtColor(latest_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            video_label.config(image=imgtk, text="")
            video_label.image = imgtk  # keep a reference!
            
            # If recording is active, write the original BGR frame to file.
            if recording and video_writer is not None:
                video_writer.write(latest_frame)
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
style.theme_use("clam")  # or use 'alt', etc.
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

model_var = tk.StringVar()
model_options = [
    "rock-paper-scissors-sxsw/14",
    "football-players-detection-3zvbc/12",
    "some-other-model-id/1"
]
model_combo = ttk.Combobox(left_frame, textvariable=model_var, values=model_options, state="readonly")
model_combo.pack(pady=(0, 20))
model_combo.current(0)  # Optionally set a default

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
