import sys
import os
import cv2
import time
import numpy as np
from datetime import datetime

from PyQt5 import QtCore, QtWidgets, QtGui

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# For Roboflow inference (replace with your actual get_model and inference code)
from inference import get_model

# For our supervision-based annotation (make sure to install supervision)
try:
    import supervision as sv
except ImportError:
    raise ImportError("Please install the supervision package (pip install supervision)")

# ------------------ Global Setup and Inference Models ------------------
ROBOFLOW_API_KEY = "tvZVhjN9hMWkURbVo84w"
PLAYER_DETECTION_MODEL_ID = "movesafep4/3"
football_model = get_model(model_id=PLAYER_DETECTION_MODEL_ID, api_key=ROBOFLOW_API_KEY)

# Dummy YOLO Pose loading (adjust as needed)
yolo_model = None  # This will be loaded later if needed

# Setup annotators (used in football inference)
ellipse_annotator = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
    thickness=2
)
football_label_annotator = sv.LabelAnnotator(
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
# Assume ball class ID is 0.
BALL_ID = 0

# Initialize a ByteTrack tracker from supervision
tracker = sv.ByteTrack()
tracker.reset()

# Global variables for collision/distance tracking
current_distance = None
collision_state = False          # True if a collision is currently active.
collision_start_time = None      # Timestamp when current collision started.
current_collision_duration = 0   # Duration (in seconds) of the ongoing collision.
collision_durations = []         # List of finalized collision durations.

# ------------------ Utility Functions ------------------
def get_video_files():
    video_folder = "videos"
    if not os.path.exists(video_folder):
        return []
    video_files = [os.path.join(video_folder, f) for f in os.listdir(video_folder)
                   if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    return video_files

# ------------------ Inference Functions ------------------
def run_football_inference(frame):
    """
    Process a frame with the football (players detection) model,
    update collision and distance information, and return an annotated frame.
    """
    global current_distance, collision_state, collision_start_time, current_collision_duration, collision_durations

    scale_factor = 0.5
    original_h, original_w = frame.shape[:2]
    small_frame = cv2.resize(frame, (int(original_w * scale_factor), int(original_h * scale_factor)))
    try:
        result = football_model.infer(small_frame, confidence=0.3)[0]
    except Exception as e:
        print(f"[ERROR] Football inference error: {e}")
        return frame

    detections = sv.Detections.from_inference(result)
    detections.xyxy = detections.xyxy / scale_factor  # scale back

    red_positions = []
    blue_positions = []
    for class_name, bbox in zip(detections['class_name'], detections.xyxy):
        if class_name == "Player Red":
            x1, y1, x2, y2 = bbox
            red_positions.append(((x1+x2)/2.0, (y1+y2)/2.0))
        elif class_name == "Player Blue":
            x1, y1, x2, y2 = bbox
            blue_positions.append(((x1+x2)/2.0, (y1+y2)/2.0))

    if red_positions and blue_positions:
        avg_red = np.mean(red_positions, axis=0)
        avg_blue = np.mean(blue_positions, axis=0)
        current_distance = np.linalg.norm(np.array(avg_red) - np.array(avg_blue))
    else:
        current_distance = None

    # --- Collision Detection Logic ---
    if current_distance is not None and current_distance < 50:
        if not collision_state:
            collision_state = True
            collision_start_time = time.time()
        current_collision_duration = time.time() - collision_start_time
    else:
        if collision_state:
            collision_durations.append(time.time() - collision_start_time)
            collision_state = False
            current_collision_duration = 0

    # Continue with annotation (dummy tracking)
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
    return annotated

def run_yolo_inference(frame):
    """
    Process a frame with the YOLO Pose model.
    (This dummy function just writes text on the frame.)
    Replace with your actual YOLO inference code.
    """
    cv2.putText(frame, "YOLO Pose Inference", (50, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
    return frame

# ------------------ Video Processor Thread ------------------
class VideoProcessor(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(QtGui.QImage)
    error_signal = QtCore.pyqtSignal(str)
    
    def __init__(self, video_source, parent=None):
        super().__init__(parent)
        self.video_source = video_source
        self._running = True
        self.capture = None
        # active_model can be None, "Players detection", or "YOLO Pose"
        self.active_model = None

    def run(self):
        self.capture = cv2.VideoCapture(self.video_source)
        if not self.capture.isOpened():
            self.error_signal.emit("Failed to open video source.")
            return
        
        prev_time = time.time()
        while self._running:
            ret, frame = self.capture.read()
            if not ret:
                self.error_signal.emit("End of video or stream error.")
                break

            # Run inference if a model is active.
            if self.active_model:
                if self.active_model == "Players detection":
                    frame = run_football_inference(frame)
                elif self.active_model == "YOLO Pose":
                    frame = run_yolo_inference(frame)

            # Compute FPS and overlay it.
            current_time = time.time()
            fps = 1.0 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
            prev_time = current_time
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Convert frame to QImage.
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            q_img = QtGui.QImage(rgb_frame.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            scaled_img = q_img.scaled(800, 600, QtCore.Qt.KeepAspectRatio)
            self.frame_ready.emit(scaled_img)
            time.sleep(0.03)
        self.capture.release()

    def stop(self):
        self._running = False
        self.wait()

    def set_active_model(self, model_name):
        self.active_model = model_name

# ------------------ Main Application Window ------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MOVE SAFE ANALYTICS")
        self.resize(1400, 900)
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; }
            QLabel, QComboBox, QPushButton { color: #ffffff; font-size: 16px; }
            QComboBox { background-color: #333333; padding: 4px; }
            QPushButton {
                background-color: #444444;
                border: 2px solid #555555;
                border-radius: 5px;
                padding: 8px;
            }
            QPushButton:hover { background-color: #555555; }
            QPushButton:pressed { background-color: #333333; }
            QPushButton:checked { background-color: #00aa00; border: 2px solid #008800; }
        """)

        # --- Main Layout ---
        central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(central_widget)
        h_layout = QtWidgets.QHBoxLayout(central_widget)
        h_layout.setContentsMargins(10, 10, 10, 10)
        h_layout.setSpacing(10)

        # --- Left Panel: Controls ---
        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(20, 20, 20, 20)
        left_layout.setSpacing(15)
        h_layout.addWidget(left_panel, 1)

        # Video selection
        label_video = QtWidgets.QLabel("Select Video:")
        left_layout.addWidget(label_video)
        self.video_combo = QtWidgets.QComboBox()
        self.update_video_combo()
        left_layout.addWidget(self.video_combo)

        # Model selection
        label_model = QtWidgets.QLabel("Select Model:")
        left_layout.addWidget(label_model)
        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.addItems(["None", "Players detection", "YOLO Pose"])
        left_layout.addWidget(self.model_combo)

        # Button to toggle video stream
        self.btn_stream = QtWidgets.QPushButton("Stream")
        self.btn_stream.setCheckable(True)
        self.btn_stream.clicked.connect(self.toggle_stream)
        left_layout.addWidget(self.btn_stream)

        # Button to toggle model inference
        self.btn_model = QtWidgets.QPushButton("Run Model")
        self.btn_model.setCheckable(True)
        self.btn_model.clicked.connect(self.toggle_model)
        left_layout.addWidget(self.btn_model)

        # Reset Graph button
        self.btn_reset_graph = QtWidgets.QPushButton("Reset Graph")
        self.btn_reset_graph.clicked.connect(self.reset_graph)
        left_layout.addWidget(self.btn_reset_graph)

        left_layout.addStretch()

        # --- Right Panel: Video Display and Graphs ---
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(10)
        h_layout.addWidget(right_panel, 3)

        # Video Display
        self.video_display = QtWidgets.QLabel("No Video", alignment=QtCore.Qt.AlignCenter)
        self.video_display.setStyleSheet("background-color: #3a3a3a; border: 1px solid #555555;")
        self.video_display.setFixedSize(800, 600)
        right_layout.addWidget(self.video_display)

        # Graphs: Use two matplotlib canvases (collision graph above distance graph)
        self.collision_fig = plt.Figure(figsize=(5,2), dpi=100)
        self.collision_canvas = FigureCanvas(self.collision_fig)
        right_layout.addWidget(self.collision_canvas)
        self.distance_fig = plt.Figure(figsize=(5,2), dpi=100)
        self.distance_canvas = FigureCanvas(self.distance_fig)
        right_layout.addWidget(self.distance_canvas)

        # Set up timers to update graphs periodically.
        self.graph_timer = QtCore.QTimer()
        self.graph_timer.timeout.connect(self.update_graphs)
        self.graph_timer.start(200)  # update every 200ms

        # ------------------ Video Processor Thread ------------------
        self.video_thread = None

    def update_video_combo(self):
        files = get_video_files()
        self.video_combo.clear()
        if files:
            self.video_combo.addItems(files)
            self.video_combo.setCurrentIndex(0)
        else:
            self.video_combo.addItem("")

    def toggle_stream(self, checked):
        if checked:
            video_file = self.video_combo.currentText()
            if video_file and os.path.exists(video_file):
                self.video_thread = VideoProcessor(video_file)
                self.video_thread.frame_ready.connect(self.update_video_display)
                self.video_thread.error_signal.connect(self.handle_video_error)
                # Set active model based on current selection.
                active_model = self.model_combo.currentText()
                if active_model == "None":
                    active_model = None
                self.video_thread.set_active_model(active_model)
                self.video_thread.start()
                self.btn_stream.setText("Stop Stream")
            else:
                QtWidgets.QMessageBox.warning(self, "Video File", "Selected video file not found.")
                self.btn_stream.setChecked(False)
        else:
            if self.video_thread:
                self.video_thread.stop()
                self.video_thread = None
            self.btn_stream.setText("Stream")
            self.video_display.setText("No Video")

    def toggle_model(self, checked):
        if self.video_thread:
            active_model = self.model_combo.currentText()
            if active_model == "None":
                active_model = None
            self.video_thread.set_active_model(active_model)
            self.btn_model.setText("Model ON" if checked else "Run Model")
        else:
            QtWidgets.QMessageBox.information(self, "No Stream", "Start the video stream first.")

    @QtCore.pyqtSlot(QtGui.QImage)
    def update_video_display(self, q_img):
        self.video_display.setPixmap(QtGui.QPixmap.fromImage(q_img))

    @QtCore.pyqtSlot(str)
    def handle_video_error(self, error_msg):
        QtWidgets.QMessageBox.critical(self, "Video Error", error_msg)
        self.btn_stream.setChecked(False)
        self.video_display.setText("No Video")

    def update_graphs(self):
        # Update distance graph
        self.distance_fig.clf()
        ax = self.distance_fig.add_subplot(111)
        if current_distance is not None:
            # Change color: red if below 50, otherwise blue.
            bar_color = 'red' if current_distance < 50 else 'blue'
            ax.bar(['Distance'], [current_distance], color=bar_color)
            ax.set_ylim(0, max(200, current_distance + 20))
            ax.set_ylabel("Distance (pixels)")
            ax.set_title("Distance between Teams")
        else:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", fontsize=12)
        self.distance_canvas.draw()

        # Update collision graph with a line plot instead of scatter
        self.collision_fig.clf()
        ax2 = self.collision_fig.add_subplot(111)
        if collision_durations:
            x_vals = list(range(1, len(collision_durations) + 1))
            ax2.plot(x_vals, collision_durations, color='red', marker='o', linestyle='-',
                    label="Finalized Collisions")
        if collision_state:
            # Plot ongoing collision as a separate point
            ax2.plot([len(collision_durations) + 1], [current_collision_duration],
                    color='orange', marker='o', linestyle='None', label="Ongoing Collision")
        ax2.set_xlabel("Collision Event")
        ax2.set_ylabel("Duration (s)")
        ax2.set_title("Collision Events (Duration)")
        handles, labels = ax2.get_legend_handles_labels()
        if handles:
            ax2.legend()
        self.collision_canvas.draw()


    def reset_graph(self):
        global collision_state, collision_start_time, current_collision_duration, collision_durations
        collision_state = False
        collision_start_time = None
        current_collision_duration = 0
        collision_durations = []
        print("[INFO] Collision graph data reset.")
        self.update_graphs()

# ------------------ Main ------------------
def show_main_window(splash):
    global main_window  # Ensure the main window reference is kept.
    main_window = MainWindow()
    main_window.show()
    splash.finish(main_window)

def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # Set up a dark palette.
    dark_palette = QtGui.QPalette()
    dark_palette.setColor(QtGui.QPalette.Window, QtGui.QColor(30, 30, 30))
    dark_palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
    dark_palette.setColor(QtGui.QPalette.Base, QtGui.QColor(45, 45, 45))
    dark_palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(66, 66, 66))
    dark_palette.setColor(QtGui.QPalette.ToolTipBase, QtCore.Qt.white)
    dark_palette.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.white)
    dark_palette.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
    dark_palette.setColor(QtGui.QPalette.Button, QtGui.QColor(45, 45, 45))
    dark_palette.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
    dark_palette.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)
    dark_palette.setColor(QtGui.QPalette.Link, QtGui.QColor(42, 130, 218))
    dark_palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(42, 130, 218))
    dark_palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.black)
    app.setPalette(dark_palette)
    
    # Attempt to load the splash image.
    splash_image_path = "images/tus_logo.jpg"
    if os.path.exists(splash_image_path):
        splash_pix = QtGui.QPixmap(splash_image_path)
    else:
        print(f"[WARNING] Splash image '{splash_image_path}' not found. Using fallback splash.")
        splash_pix = QtGui.QPixmap(800, 600)
        splash_pix.fill(QtGui.QColor("darkGray"))
        painter = QtGui.QPainter(splash_pix)
        painter.setPen(QtCore.Qt.white)
        font = QtGui.QFont("Arial", 24)
        painter.setFont(font)
        painter.drawText(splash_pix.rect(), QtCore.Qt.AlignCenter, "MSA - Move Safe Analytic")
        painter.end()
    
    if splash_pix.isNull():
        print("[ERROR] Failed to load splash image and fallback pixmap is null.")
        sys.exit(1)
    
    # Scale the splash image.
    scaled_pix = splash_pix.scaled(800, 600, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
    splash = QtWidgets.QSplashScreen(scaled_pix)
    splash.setMask(scaled_pix.mask())
    splash.showMessage("MSA - Move Safe Analytic", 
                       QtCore.Qt.AlignBottom | QtCore.Qt.AlignCenter, 
                       QtCore.Qt.white)
    splash.show()
    app.processEvents()
    
    # Delay for 3 seconds, then show the main window.
    QtCore.QTimer.singleShot(3000, lambda: show_main_window(splash))
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
