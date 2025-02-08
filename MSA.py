import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk

def on_show_stream():
    print("[INFO] Show Stream clicked.")
    # TODO: Add your logic to start RTMP capture, etc.

def on_stop_stream():
    print("[INFO] Stop Stream clicked.")
    # TODO: Add your logic to stop RTMP capture.

def on_run_model():
    print("[INFO] Start Model clicked.")
    # TODO: Add your logic to run inference pipeline.

def on_stop_model():
    print("[INFO] Stop Model clicked.")
    # TODO: Add your logic to stop inference pipeline.

def on_start_recording():
    print("[INFO] Start Recording clicked.")
    # TODO: Add your logic to start video recording.

def on_stop_recording():
    print("[INFO] Stop Recording clicked.")
    # TODO: Add your logic to stop video recording.

def on_escape(event):
    """
    By default, we bind the ESC key to exit fullscreen or close the app.
    If you want to only exit fullscreen (but not close),
    use: root.attributes("-fullscreen", False) instead of root.quit().
    """
    root.quit()

##############################
# Main App
##############################
root = tk.Tk()
root.title("Full-Screen Dark UI")

# 1) Make it full-screen
root.attributes("-fullscreen", True)

# 2) Dark theme styling with Ttk
style = ttk.Style()
# 'clam' or 'alt' can be easier to customize 
style.theme_use("clam")

# General background
root.configure(bg="black")

# Customize TFrame, TButton, TLabel, etc.
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

# Bind ESC to quit (or turn off fullscreen)
root.bind("<Escape>", on_escape)

##############################
# Layout Frames
##############################

# We’ll have a left frame (menu) and a right frame (display area).
left_frame = ttk.Frame(root, width=300)
left_frame.pack(side=tk.LEFT, fill=tk.Y)

# The right frame will hold two sub-frames: top (video) and bottom (placeholder).
right_frame = ttk.Frame(root)
right_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

# Top frame for the video area
top_frame = ttk.Frame(right_frame)
top_frame.pack(side=tk.TOP, expand=True, fill=tk.BOTH)

# Bottom frame for placeholders/future content
bottom_frame = ttk.Frame(right_frame, height=200)
bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)


##############################
# Left Menu Panel
##############################
# 1) RTMP / URL input
url_label = ttk.Label(left_frame, text="RTMP URL:")
url_label.pack(pady=(20, 5))

url_entry = ttk.Entry(left_frame, width=25)
url_entry.insert(0, "rtmp://127.0.0.1:1935/stream")
url_entry.pack(pady=(0, 20))

# 2) Show/Stop Stream
stream_button_frame = ttk.Frame(left_frame)
stream_button_frame.pack(pady=10)

btn_show_stream = ttk.Button(stream_button_frame, text="Show Stream", command=on_show_stream)
btn_show_stream.grid(row=0, column=0, padx=5)

btn_stop_stream = ttk.Button(stream_button_frame, text="Stop Stream", command=on_stop_stream)
btn_stop_stream.grid(row=0, column=1, padx=5)

# 3) Model dropdown
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

# 4) Start/Stop Model
model_button_frame = ttk.Frame(left_frame)
model_button_frame.pack(pady=10)

btn_run_model = ttk.Button(model_button_frame, text="Start Model", command=on_run_model)
btn_run_model.grid(row=0, column=0, padx=5)

btn_stop_model = ttk.Button(model_button_frame, text="Stop Model", command=on_stop_model)
btn_stop_model.grid(row=0, column=1, padx=5)

# 5) Start/Stop Recording
record_button_frame = ttk.Frame(left_frame)
record_button_frame.pack(pady=10)

btn_start_recording = ttk.Button(record_button_frame, text="Start Recording", command=on_start_recording)
btn_start_recording.grid(row=0, column=0, padx=5)

btn_stop_recording = ttk.Button(record_button_frame, text="Stop Recording", command=on_stop_recording)
btn_stop_recording.grid(row=0, column=1, padx=5)


##############################
# Right Side: Video Display (Top) and Future Area (Bottom)
##############################

# For the top display, let's just put a label for now. 
# Later, you can place an ImageTk label (for live video) or any other widget(s).
video_label = tk.Label(top_frame, text="No Video", bg="black", fg="white")
video_label.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

# Bottom placeholder
bottom_label = ttk.Label(bottom_frame, text="Placeholder for future content.")
bottom_label.pack(padx=10, pady=10)

##############################
# MAINLOOP
##############################
root.mainloop()
