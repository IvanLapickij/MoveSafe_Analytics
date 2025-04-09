import threading
import cv2
import time
from datetime import datetime
from PyQt5 import QtCore, QtGui
from threading import Lock

class FrameRateController:
    def __init__(self, target_fps):
        self.interval = 1.0 / target_fps
        self.last_display_time = 0

    def should_display(self):
        now = time.time()
        if now - self.last_display_time >= self.interval:
            self.last_display_time = now
            return True
        return False


class VideoProcessor(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(QtGui.QImage)
    error_signal = QtCore.pyqtSignal(str)

    def __init__(self, video_source, inference_model=None, parent=None):
        super().__init__(parent)
        self.video_source = video_source
        self.inference_model = inference_model
        self._running = True
        self.capture = None
        self.recording = False
        self.video_writer = None
        self.model_lock = Lock()

    def set_recording(self, record: bool):
        self.recording = record

    def run(self):
        self.capture = cv2.VideoCapture(self.video_source)
        if not self.capture.isOpened():
            self.error_signal.emit(f"Failed to open video source:\n{self.video_source}")
            return

        self.fps_controller = FrameRateController(target_fps=30)

        frame_counter = 0
        process_every = 2
        prev_time = time.time()
        last_tracker_log_time = 0

        annotated_frame = None

        while self._running:
            ret, frame = self.capture.read()
            if not ret:
                self.error_signal.emit("End of video or stream error.")
                break

            frame_counter += 1

            with self.model_lock:
                model = self.inference_model

            annotated_frame = frame.copy()

            if model and frame_counter % process_every == 0:
                try:
                    annotated_frame = model.predict(frame)
                    if hasattr(model, 'collision_tracker') and model.collision_tracker:
                        now = time.time()
                        if now - last_tracker_log_time >= 1.0:
                            last_tracker_log_time = now
                            dist = model.collision_tracker.current_distance
                            if dist is not None:
                                print(f"[TRACKER] Distance: {dist:.2f}")
                except Exception as e:
                    print(f"[ERROR] Inference failed: {e}")

            # FPS
            current_time = time.time()
            fps = 1.0 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
            prev_time = current_time
            cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            if self.recording:
                if self.video_writer is None:
                    h, w = frame.shape[:2]
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"recordings/recording_{timestamp}.avi"
                    fourcc = cv2.VideoWriter_fourcc(*"XVID")
                    self.video_writer = cv2.VideoWriter(filename, fourcc, 30, (w, h))
                self.video_writer.write(annotated_frame)
            elif self.video_writer:
                self.video_writer.release()
                self.video_writer = None

            # 🔥 Real-time controlled GUI updates
            if self.fps_controller.should_display():
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                q_img = QtGui.QImage(rgb_frame.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
                scaled_img = q_img.scaled(800, 600, QtCore.Qt.KeepAspectRatio)
                self.frame_ready.emit(scaled_img)

        self.capture.release()



    def stop(self):
        self._running = False
        self.wait()

    def set_inference_model(self, model):
        with self.model_lock:
            self.inference_model = model

    def set_model(self, model):
        with self.model_lock:
            self.inference_model = model
