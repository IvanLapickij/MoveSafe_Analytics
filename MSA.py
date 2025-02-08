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

#############################
# Main Functions
#############################
def on_show_stream():
    """
    Called when 'Show Stream' button is clicked.
    1) Stop any existing RTMP thread or pipeline.
    2) Start a new thread that captures from RTMP.
    """
    global stop_rtmp_flag, rtmp_thread
    
    print("[INFO] 'Show Stream' button clicked.")

    rtmp_url = url_entry.get()
    if not rtmp_url:
        messagebox.showwarning("No RTMP URL", "Please enter an RTMP URL.")
        return

    # Stop any running pipeline or old RTMP capture.
    on_stop_model()
    stop_rtmp()

    stop_rtmp_flag = False
    def rtmp_loop():
        print(f"[INFO] rtmp_loop started with URL = {rtmp_url}")
        cap = cv2.VideoCapture(rtmp_url)

        while not stop_rtmp_flag:
            ret, frame = cap.read()
            if not ret:
                break
            global latest_frame
            latest_frame = frame
            time.sleep(0.03)  # ~30 fps

        cap.release()
        print("[INFO] RTMP capture loop ended.")

    rtmp_thread = threading.Thread(target=rtmp_loop, daemon=True)
    rtmp_thread.start()
    print("[INFO] RTMP thread started.")


def on_stop_stream():
    print("[INFO] Stop Stream clicked.")
    stop_rtmp()

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


def on_run_model():
    """
    Called when 'Start Model' button is clicked.
    """
    print("[INFO] Start Model clicked.")
    # Example logic:
    # 1) Optionally stop RTMP so only pipeline runs
    stop_rtmp()
    # 2) stop_model() to ensure old pipeline isn't running
    on_stop_model()
    # 3) Start pipeline
    start_pipeline_inference()


def on_stop_model():
    print("[INFO] Stop Model clicked.")
    stop_model()

def stop_model():
    """
    Stops the inference pipeline if running.
    """
    global stop_pipeline_flag, pipeline_thread, latest_frame
    print("[INFO] stop_model() called.")
    stop_pipeline_flag = True

    if pipeline_thread and pipeline_thread.is_alive():
        pipeline_thread.join()
        pipeline_thread = None
        print("[INFO] Pipeline thread joined successfully.")

    # Clear frame so the GUI can show "No Video"
    latest_frame = None
    video_label.config(image="", text="No Video", fg="white")


def on_start_recording():
    print("[INFO] Start Recording clicked.")
    # TODO: Add your logic to start video recording (create VideoWriter, etc.)

def on_stop_recording():
    print("[INFO] Stop Recording clicked.")
    # TODO: Add your logic to stop video recording (release VideoWriter, etc.)

def on_escape(event):
    """
    Bind ESC to quit the app (or exit fullscreen).
    """
    root.quit()


#############################
# Pipeline Placeholder
#############################
def start_pipeline_inference():
    global pipeline_thread, stop_pipeline_flag
    print("[INFO] start_pipeline_inference called.")

    # Ensure no old pipeline
    stop_model()

    stop_pipeline_flag = False

    def pipeline_loop():
        print("[INFO] Pipeline loop started.")
        # Example logic for pipeline:
        # while not stop_pipeline_flag:
        #     # do inference
        #     # update latest_frame
        #     time.sleep(0.1)
        while not stop_pipeline_flag:
            time.sleep(0.1)
        print("[INFO] Pipeline loop ended.")

    pipeline_thread = threading.Thread(target=pipeline_loop, daemon=True)
    pipeline_thread.start()
    print("[INFO] Pipeline thread started.")


#############################
# Update Frame (display video)
#############################
def update_frame():
    global latest_frame
    if latest_frame is None:
        video_label.config(text="No Video", image="")
    else:
        try:
            frame_rgb = cv2.cvtColor(latest_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            video_label.config(image=imgtk, text="")
            video_label.image = imgtk
        except Exception as e:
            print("[ERROR] Could not display frame:", e)
            video_label.config(image="", text="No Video")

    root.after(50, update_frame)


#############################
# Build the GUI
#############################

root = tk.Tk()
root.title("Full-Screen Dark UI")

# Make it full-screen
root.attributes("-fullscreen", True)

# Dark theme styling with Ttk
style = ttk.Style()
style.theme_use("clam")  # or 'alt', etc.
root.configure(bg="black")

style.configure("TFrame", background="black")
style.configure(
    "TButton",
    background="gray25",
    foreground="white",
    bordercolor="gray25",
    focusthickness=3,
    focuscolor="gray25"
)
style.configure(
    "TLabel",
    background="black",
    foreground="white"
)
style.configure(
    "TCombobox",
    fieldbackground="gray25",
    foreground="white"
)
style.configure(
    "TEntry",
    fieldbackground="gray25",
    foreground="white"
)
style.map(
    "TButton",
    foreground=[("active", "white"), ("disabled", "grey")],
    background=[("active", "gray40"), ("disabled", "gray20")]
)

root.bind("<Escape>", on_escape)

# Left/Right frames
left_frame = ttk.Frame(root, width=300)
left_frame.pack(side=tk.LEFT, fill=tk.Y)

right_frame = ttk.Frame(root)
right_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

# Top/bottom split in right frame
top_frame = ttk.Frame(right_frame)
top_frame.pack(side=tk.TOP, expand=True, fill=tk.BOTH)

bottom_frame = ttk.Frame(right_frame, height=200)
bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)

# Left Menu Panel
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

# Right Side: Video display up top
video_label = tk.Label(top_frame, text="No Video", bg="black", fg="white")
video_label.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

# Placeholder at bottom
bottom_label = ttk.Label(bottom_frame, text="Placeholder for future content.")
bottom_label.pack(padx=10, pady=10)

# Kick off frame updates
update_frame()

# Start Mainloop
root.mainloop()