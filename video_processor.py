import cv2
import time
from datetime import datetime
from PyQt5 import QtCore, QtGui
from threading import Lock


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

        # 🔒 Add this to support safe model switching
        self.model_lock = Lock()


    def set_recording(self, record: bool):
        self.recording = record

    def run(self):
        self.capture = cv2.VideoCapture(self.video_source)
        if not self.capture.isOpened():
            self.error_signal.emit(f"Failed to open video source:\n{self.video_source}")
            return


        prev_time = time.time()
        while self._running:
            ret, frame = self.capture.read()
            if not ret:
                self.error_signal.emit("End of video or stream error.")
                break

            with self.model_lock:
                model = self.inference_model

            if model:
                frame = model.predict(frame)


            current_time = time.time()
            fps = 1.0 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
            prev_time = current_time
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            if self.recording:
                if self.video_writer is None:
                    h, w = frame.shape[:2]
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"recordings/recording_{timestamp}.avi"
                    fourcc = cv2.VideoWriter_fourcc(*"XVID")
                    self.video_writer = cv2.VideoWriter(filename, fourcc, 30, (w, h))
                self.video_writer.write(frame)
            elif self.video_writer:
                self.video_writer.release()
                self.video_writer = None

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            q_img = QtGui.QImage(rgb_frame.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
            scaled_img = q_img.scaled(800, 600, QtCore.Qt.KeepAspectRatio)
            self.frame_ready.emit(scaled_img)
            time.sleep(0.03)

        self.capture.release()

    def stop(self):
        self._running = False
        self.wait()

    def set_inference_model(self, model):
        self.inference_model = model
        
    def set_model(self, model):
        with self.model_lock:
            self.inference_model = model

