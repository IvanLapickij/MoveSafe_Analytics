# ui/main_window.py
import datetime
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtCore, QtWidgets, QtGui
import csv
from config import VIDEO_FOLDER, SPLASH_IMAGE, VIDEO_WIDTH, VIDEO_HEIGHT
from video_processor import VideoProcessor
from collision_tracker import CollisionTracker
from inference import get_model
from inference_models.football_detector import FootballDetector
import supervision as sv
from ultralytics import YOLO
from inference_models.pose_detector import PoseDetector  # If you've modularized it
pose_model = YOLO("yolo11n-pose.pt")  # trained model path

class PoseDetector:
    def __init__(self, model):
        self.model = model

    def predict(self, frame):
        try:
            results = self.model.predict(frame, conf=0.5)
            return results[0].plot()
        except Exception as e:
            print(f"[YOLO Pose Error] {e}")
            return frame

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MOVE SAFE ANALYTICS")
        self.resize(1400, 900)
        self.video_thread = None
        self.collision_tracker = CollisionTracker()

        self.init_ui()
        self.graph_timer = QtCore.QTimer()
        self.graph_timer.timeout.connect(self.update_graphs)
        self.graph_timer.start(200)
    
    def export_logs(self):
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"collision_log_{now}.csv"

        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Collision #", "Duration (s)"])
            for i, duration in enumerate(self.collision_tracker.collision_durations, 1):
                writer.writerow([i, round(duration, 2)])
            if self.collision_tracker.collision_state:
                writer.writerow(["Ongoing", round(self.collision_tracker.current_collision_duration, 2)])

        QtWidgets.QMessageBox.information(self, "Export Complete", f"Saved to: {filepath}")

    def init_ui(self):
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
        central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(central_widget)
        h_layout = QtWidgets.QHBoxLayout(central_widget)

        # Left Panel
        left_panel = QtWidgets.QVBoxLayout()
        self.video_combo = QtWidgets.QComboBox()
        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.addItems(["None", "Players detection", "YOLO Pose"])
        self.btn_stream = QtWidgets.QPushButton("Stream")
        self.btn_model = QtWidgets.QPushButton("Run Model")
        self.btn_record = QtWidgets.QPushButton("Record Video")
        self.btn_reset_graph = QtWidgets.QPushButton("Reset Graph")
        # Export logs
        self.btn_export_csv = QtWidgets.QPushButton("Export Logs (CSV)")
        self.btn_export_csv.clicked.connect(self.export_logs)
        


        self.btn_stream.setCheckable(True)
        self.btn_model.setCheckable(True)
        self.btn_record.setCheckable(True)

        self.btn_stream.clicked.connect(self.toggle_stream)
        self.btn_model.clicked.connect(self.toggle_model)
        self.btn_record.clicked.connect(self.toggle_recording)
        self.btn_reset_graph.clicked.connect(self.reset_graph)
        

        left_panel.addWidget(QtWidgets.QLabel("Select Video:"))
        left_panel.addWidget(self.video_combo)
        left_panel.addWidget(QtWidgets.QLabel("Select Model:"))
        left_panel.addWidget(self.model_combo)
        left_panel.addWidget(self.btn_stream)
        left_panel.addWidget(self.btn_model)
        left_panel.addWidget(self.btn_record)
        left_panel.addWidget(self.btn_reset_graph)
        left_panel.addWidget(self.btn_export_csv)
        left_panel.addStretch(1)

        self.distance_fig = plt.Figure(figsize=(5, 2), dpi=100)
        self.distance_canvas = FigureCanvas(self.distance_fig)
        left_panel.addWidget(self.distance_canvas)

        # Right Panel
        right_panel = QtWidgets.QVBoxLayout()
        self.video_display = QtWidgets.QLabel("No Video", alignment=QtCore.Qt.AlignCenter)
        self.video_display.setFixedSize(VIDEO_WIDTH, VIDEO_HEIGHT)
        self.video_display.setStyleSheet("background-color: #3a3a3a; border: 1px solid #555555;")
        self.collision_fig = plt.Figure(figsize=(5, 2), dpi=100)
        self.collision_canvas = FigureCanvas(self.collision_fig)

        right_panel.addWidget(self.video_display)
        right_panel.addWidget(self.collision_canvas)

        h_layout.addLayout(left_panel, 1)
        h_layout.addLayout(right_panel, 3)

        self.update_video_combo()

    def update_video_combo(self):
        self.video_combo.clear()
        files = [f for f in os.listdir(VIDEO_FOLDER)
                 if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
        self.video_combo.addItems([os.path.join(VIDEO_FOLDER, f) for f in files])
        self.video_combo.addItem("Enter stream URL...")

    def toggle_stream(self, checked):
        if checked:
            # 🔁 Stop any existing thread BEFORE starting a new one
            if self.video_thread:
                self.video_thread.stop()
                self.video_thread = None
                self.video_display.clear()  # Clear the QLabel

            source = self.video_combo.currentText()
            if source == "Enter stream URL...":
                url, ok = QtWidgets.QInputDialog.getText(self, "Stream URL", "Enter stream URL:")
                if not ok or not url:
                    self.btn_stream.setChecked(False)
                    return
                source = url

            if not os.path.exists(source) and not (
                source.startswith("rtmp://") or
                source.startswith("rtsp://") or
                source.startswith("http://") or
                source.startswith("https://")
            ):

                QtWidgets.QMessageBox.warning(self, "File not found", "Video file not found.")
                self.btn_stream.setChecked(False)
                return

            model = None
            if self.model_combo.currentText() == "Players detection":
                rf_model = get_model(model_id="movesafep4/3", api_key="tvZVhjN9hMWkURbVo84w")
                tracker = sv.ByteTrack()
                tracker.reset()
                model = FootballDetector(rf_model, tracker, self.collision_tracker)

            self.video_thread = VideoProcessor(source, model)
            self.video_thread.frame_ready.connect(self.update_video_display)
            self.video_thread.error_signal.connect(self.show_error)
            self.video_thread.start()
            self.btn_stream.setText("Stop Stream")
        else:
            if self.video_thread:
                self.video_thread.stop()
                self.video_thread = None
            self.btn_stream.setText("Stream")
            self.video_display.setText("No Video")


    def toggle_model(self, checked):
        if not self.video_thread:
            QtWidgets.QMessageBox.information(self, "No Stream", "Start a stream first.")
            self.btn_model.setChecked(False)
            return

        if checked:
            selected_model = self.model_combo.currentText()
            model = None

            if selected_model == "Players detection":
                rf_model = get_model(model_id="movesafep4/3", api_key="tvZVhjN9hMWkURbVo84w")
                tracker = sv.ByteTrack()
                tracker.reset()
                model = FootballDetector(rf_model, tracker, self.collision_tracker)
            elif selected_model == "YOLO Pose":
                model = PoseDetector(pose_model)


            if model:
                self.video_thread.set_model(model)
                self.btn_model.setText("Model ON")
                self.btn_model.setStyleSheet("background-color: #00aa00; color: white;")
        else:
            self.video_thread.set_model(None)
            self.btn_model.setText("Run Model")
            self.btn_model.setStyleSheet("")




    def toggle_recording(self, checked):
        if self.video_thread:
            self.video_thread.set_recording(checked)
            self.btn_record.setText("Stop Recording" if checked else "Record Video")
        else:
            QtWidgets.QMessageBox.information(self, "No Stream", "Start a stream first.")
            self.btn_record.setChecked(False)

    def reset_graph(self):
        self.collision_tracker.reset()
        self.update_graphs()

    def update_video_display(self, q_img):
        self.video_display.setPixmap(QtGui.QPixmap.fromImage(q_img))

    def show_error(self, msg):
        QtWidgets.QMessageBox.critical(self, "Error", msg)

    def update_graphs(self):
        # Distance Graph
        self.distance_fig.clf()
        ax = self.distance_fig.add_subplot(111)
        if self.collision_tracker.current_distance is not None:
            color = 'red' if self.collision_tracker.current_distance < 50 else 'blue'
            ax.bar(['Distance'], [self.collision_tracker.current_distance], color=color)
            ax.set_ylim(0, max(200, self.collision_tracker.current_distance + 20))
            ax.set_ylabel("Distance (pixels)")
            ax.set_title("Distance between Players")
        else:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
        self.distance_canvas.draw()

        # Collision Duration Graph
        self.collision_fig.clf()
        ax2 = self.collision_fig.add_subplot(111)
        if self.collision_tracker.collision_durations:
            x_vals = list(range(1, len(self.collision_tracker.collision_durations) + 1))
            ax2.plot(x_vals, self.collision_tracker.collision_durations, 'ro-', label="Finalized")
        if self.collision_tracker.collision_state:
            ax2.plot([len(self.collision_tracker.collision_durations) + 1],
                    [self.collision_tracker.current_collision_duration], 'yo', label="Ongoing")
        ax2.set_title("Collision Events (Duration)")
        ax2.set_xlabel("Event")
        ax2.set_ylabel("Seconds")
        if ax2.get_legend_handles_labels()[0]:
            ax2.legend()
        self.collision_canvas.draw()
    


def show_splash_screen(main_window):
    if os.path.exists(SPLASH_IMAGE):
        pixmap = QtGui.QPixmap(SPLASH_IMAGE)
    else:
        pixmap = QtGui.QPixmap(800, 600)
        pixmap.fill(QtGui.QColor("darkGray"))
        painter = QtGui.QPainter(pixmap)
        painter.setPen(QtCore.Qt.white)
        painter.setFont(QtGui.QFont("Arial", 24))
        painter.drawText(pixmap.rect(), QtCore.Qt.AlignCenter, "MSA - Move Safe Analytic")
        painter.end()

    splash = QtWidgets.QSplashScreen(pixmap)
    splash.showMessage("MSA - Move Safe Analytic", QtCore.Qt.AlignBottom | QtCore.Qt.AlignCenter, QtCore.Qt.white)
    splash.show()
    QtWidgets.QApplication.processEvents()
    QtCore.QTimer.singleShot(3000, lambda: (main_window.show(), splash.finish(main_window)))
    return splash



