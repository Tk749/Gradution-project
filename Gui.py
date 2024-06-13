import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QLabel, QVBoxLayout, QWidget, QHBoxLayout, QSlider, QSizePolicy, QMessageBox, QStackedWidget, QSpacerItem, QSizePolicy
from PyQt5.QtGui import QFont, QIcon, QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
import cv2
from ultralytics import YOLO
import os
import time

class ClickableLabel(QLabel):
    clicked = pyqtSignal()

    def mousePressEvent(self, event):
        self.clicked.emit()

class VideoProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("YOLO Video Processing")
        self.setGeometry(100, 100, 1280, 720)
        self.setStyleSheet("background-color: #f5f5f5;")
        self.setWindowIcon(QIcon("logo.png"))

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.cap = None
        self.model = None
        self.video_loaded = False
        self.model_loaded = False
        self.total_frames = 0
        self.is_seeking = False
        self.processed_frames = 0
        self.processed_video = []
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self.playback_frame_index)
        self.playback_index = 0
        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self.update_camera_frame)

        self.initUI()

        # FPS hesaplama için değişkenler
        self.start_time = None
        self.frame_counter = 0

    def initUI(self):
        # Stacked Widget to switch between pages
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        # First Page - Loading Video and Model
        self.first_page = QWidget()
        self.init_first_page()
        self.stacked_widget.addWidget(self.first_page)

        # Second Page - Video Processing
        self.second_page = QWidget()
        self.init_second_page()
        self.stacked_widget.addWidget(self.second_page)

        # Third Page - Camera Input Processing
        self.third_page = QWidget()
        self.init_third_page()
        self.stacked_widget.addWidget(self.third_page)

    def init_first_page(self):
        # Label
        self.label = QLabel("Load Video and Model", self.first_page)
        self.label.setFont(QFont("Arial", 16, QFont.Bold))
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("color: #333333; margin-bottom: 20px;")

        # Buttons
        self.video_button = QPushButton("Select Video", self.first_page)
        self.video_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.video_button.setStyleSheet("""
            QPushButton {
                background-color: #FF9800; 
                color: white; 
                border: none; 
                padding: 10px 20px; 
                font-size: 14px; 
                margin: 10px;
                cursor: pointer; 
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #FB8C00;
            }
        """)
        self.video_button.clicked.connect(self.openFileNameDialog)

        self.video_path_label = QLabel("Selected Video: None", self.first_page)
        self.video_path_label.setFont(QFont("Arial", 12))
        self.video_path_label.setAlignment(Qt.AlignCenter)
        self.video_path_label.setStyleSheet("color: #333333; margin-bottom: 10px;")

        self.model_button = QPushButton("Select Model", self.first_page)
        self.model_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.model_button.setStyleSheet("""
            QPushButton {
                background-color: #FF9800; 
                color: white; 
                border: none; 
                padding: 10px 20px; 
                font-size: 14px; 
                margin: 10px;
                cursor: pointer; 
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #FB8C00;
            }
        """)
        self.model_button.clicked.connect(self.openModelDialog)

        self.model_path_label = QLabel("Selected Model: None", self.first_page)
        self.model_path_label.setFont(QFont("Arial", 12))
        self.model_path_label.setAlignment(Qt.AlignCenter)
        self.model_path_label.setStyleSheet("color: #333333; margin-bottom: 20px;")

        self.proceed_button = QPushButton("Go to Processing Page", self.first_page)
        self.proceed_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.proceed_button.setStyleSheet("""
            QPushButton {
                background-color: #d3d3d3; 
                color: #808080; 
                border: none; 
                padding: 10px 20px; 
                font-size: 14px; 
                margin: 10px;
                cursor: not-allowed; 
                border-radius: 8px;
            }
            QPushButton:enabled {
                background-color: #FF9800; 
                color: white; 
                cursor: pointer;
            }
            QPushButton:enabled:hover {
                background-color: #FB8C00;
            }
        """)
        self.proceed_button.setEnabled(False)
        self.proceed_button.clicked.connect(self.go_to_processing_page)

        self.camera_button = QPushButton("Go to Camera Processing Page", self.first_page)
        self.camera_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.camera_button.setStyleSheet("""
            QPushButton {
                background-color: #d3d3d3; 
                color: #808080; 
                border: none; 
                padding: 10px 20px; 
                font-size: 14px; 
                margin: 10px;
                cursor: not-allowed; 
                border-radius: 8px;
            }
            QPushButton:enabled {
                background-color: #FF9800; 
                color: white; 
                cursor: pointer;
            }
            QPushButton:enabled:hover {
                background-color: #FB8C00;
            }
        """)
        self.camera_button.setEnabled(False)
        self.camera_button.clicked.connect(self.go_to_camera_page)

        # Layouts
        button_layout = QVBoxLayout()
        button_layout.setAlignment(Qt.AlignCenter)
        button_layout.addWidget(self.video_button)
        button_layout.addWidget(self.video_path_label)
        button_layout.addWidget(self.model_button)
        button_layout.addWidget(self.model_path_label)
        button_layout.addWidget(self.proceed_button)
        button_layout.addWidget(self.camera_button)

        main_layout = QVBoxLayout(self.first_page)
        main_layout.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.label)
        main_layout.addLayout(button_layout)
        main_layout.setContentsMargins(20, 20, 20, 20)

    def init_second_page(self):
        # Video Display Label
        self.video_label = ClickableLabel(self.second_page)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Frame Information Labels
        self.frame_info_label = QLabel(self.second_page)
        self.frame_info_label.setFont(QFont("Arial", 12))
        self.frame_info_label.setAlignment(Qt.AlignCenter)
        self.frame_info_label.setStyleSheet("color: #333333; margin: 10px;")

        # Detection Results Text Area
        self.detection_results_text = QLabel(self.second_page)
        self.detection_results_text.setFont(QFont("Arial", 10))
        self.detection_results_text.setAlignment(Qt.AlignTop)
        self.detection_results_text.setWordWrap(True)
        self.detection_results_text.setStyleSheet("color: #333333; margin: 10px; border: 1px solid #ccc; padding: 5px;")
        self.detection_results_text.setFixedHeight(40)  # Set a smaller fixed height

        # Seek Bar
        self.seek_bar = QSlider(Qt.Horizontal, self.second_page)
        self.seek_bar.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: #FF9800;
                margin: 2px 0;
            }

            QSlider::handle:horizontal {
                background: #FFFFFF;
                border: 1px solid #666666;
                width: 18px;
                height: 18px;
                margin: -5px 0; 
                border-radius: 9px;
            }

            QSlider::sub-page:horizontal {
                background: #FB8C00;
                border: 1px solid #999999;
                height: 8px;
                margin: 2px 0;
            }
        """)
        self.seek_bar.setRange(0, 100)
        self.seek_bar.sliderPressed.connect(self.start_seeking)
        self.seek_bar.sliderReleased.connect(self.end_seeking)
        self.seek_bar.sliderMoved.connect(self.seek_video)

        # Play Again Button
        self.play_again_button = QPushButton("Play Again", self.second_page)
        self.play_again_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.play_again_button.setStyleSheet("""
            QPushButton {
                background-color: #d3d3d3; 
                color: #808080; 
                border: none; 
                padding: 10px 20px; 
                font-size: 14px; 
                cursor: not-allowed; 
                border-radius: 8px;
            }
            QPushButton:enabled {
                background-color: #4CAF50; 
                color: white; 
                cursor: pointer;
            }
            QPushButton:enabled:hover {
                background-color: #45a049;
            }
        """)
        self.play_again_button.setEnabled(False)
        self.play_again_button.clicked.connect(self.play_processed_video)

        # Download Button
        self.download_button = QPushButton("Download", self.second_page)
        self.download_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.download_button.setStyleSheet("""
            QPushButton {
                background-color: #d3d3d3; 
                color: #808080; 
                border: none; 
                padding: 10px 20px; 
                font-size: 14px; 
                cursor: not-allowed; 
                border-radius: 8px;
            }
            QPushButton:enabled {
                background-color: #2196F3; 
                color: white; 
                cursor: pointer;
            }
            QPushButton:enabled:hover {
                background-color: #1976D2;
            }
        """)
        self.download_button.setEnabled(False)
        self.download_button.clicked.connect(self.download_processed_video)

        # Back Button
        self.back_button = QPushButton("Back", self.second_page)
        self.back_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.back_button.setIcon(QIcon("back_icon.png"))
        self.back_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336; 
                color: white; 
                border: none; 
                padding: 10px 20px; 
                font-size: 14px; 
                cursor: pointer; 
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        self.back_button.clicked.connect(self.back_to_loading)

        # Layout
        video_layout = QVBoxLayout()
        video_layout.addWidget(self.video_label)
        video_layout.addWidget(self.frame_info_label)
        video_layout.addWidget(self.detection_results_text)
        video_layout.addWidget(self.seek_bar)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.play_again_button)
        button_layout.addWidget(self.download_button)
        button_layout.addWidget(self.back_button)
        button_layout.addStretch()

        main_layout = QVBoxLayout(self.second_page)
        main_layout.addLayout(video_layout)
        main_layout.addLayout(button_layout)
        main_layout.setContentsMargins(20, 20, 20, 20)

    def init_third_page(self):
        self.camera_view = QLabel(self.third_page)
        self.camera_view.setAlignment(Qt.AlignCenter)
        self.camera_view.setStyleSheet("background-color: black;")
        self.camera_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.detection_results_camera = QLabel(self.third_page)
        self.detection_results_camera.setFont(QFont("Arial", 10))
        self.detection_results_camera.setAlignment(Qt.AlignTop)
        self.detection_results_camera.setWordWrap(True)
        self.detection_results_camera.setStyleSheet("color: #333333; margin: 10px;")
        self.detection_results_camera.setFixedHeight(200)  # Adjust as needed

        self.camera_start_button = QPushButton("Start", self.third_page)
        self.camera_start_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.camera_start_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50; 
                color: white; 
                border: none; 
                padding: 10px 20px; 
                font-size: 14px; 
                cursor: pointer; 
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.camera_start_button.clicked.connect(self.start_camera)

        self.camera_stop_button = QPushButton("Stop", self.third_page)
        self.camera_stop_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.camera_stop_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336; 
                color: white; 
                border: none; 
                padding: 10px 20px; 
                font-size: 14px; 
                cursor: pointer; 
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        self.camera_stop_button.setEnabled(False)
        self.camera_stop_button.clicked.connect(self.stop_camera)

        self.camera_back_button = QPushButton("Back", self.third_page)
        self.camera_back_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.camera_back_button.setIcon(QIcon("back_icon.png"))
        self.camera_back_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336; 
                color: white; 
                border: none; 
                padding: 10px 20px; 
                font-size: 14px; 
                cursor: pointer; 
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        self.camera_back_button.clicked.connect(self.back_to_loading_from_camera)

        # Layout
        video_layout = QVBoxLayout()
        video_layout.addWidget(self.camera_view)

        detection_layout = QVBoxLayout()
        detection_layout.addWidget(self.detection_results_camera)

        button_layout = QVBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.camera_start_button)
        button_layout.addWidget(self.camera_stop_button)
        button_layout.addWidget(self.camera_back_button)

        side_layout = QVBoxLayout()
        side_layout.addLayout(detection_layout)
        side_layout.addLayout(button_layout)

        main_layout = QHBoxLayout(self.third_page)
        main_layout.addLayout(video_layout, 2)
        main_layout.addLayout(side_layout, 1)
        main_layout.setContentsMargins(20, 20, 20, 20)

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        fileName, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "All Videos (*.mp4 *.avi *.mov);;All Files (*)", options=options)
        if fileName:
            self.video_path_label.setText(f"Selected Video: {fileName}")
            self.load_video(fileName)

    def openModelDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        modelFileName, _ = QFileDialog.getOpenFileName(self, "Select Model", "", "YOLO Models (*.pt);;All Files (*)", options=options)
        if modelFileName:
            self.model_path_label.setText(f"Selected Model: {modelFileName}")
            self.load_model(modelFileName)

    def load_model(self, modelFileName):
        try:
            self.model = YOLO(modelFileName)
            self.model_loaded = True
            self.check_ready_to_proceed()
        except Exception as e:
            QMessageBox.critical(self, "Model Loading Error", f"An error occurred while loading the model: {str(e)}")

    def load_video(self, fileName):
        self.cap = cv2.VideoCapture(fileName)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Video Loading Error", "Unable to load video. Please select a valid video file.")
            return
        self.video_loaded = True
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.seek_bar.setRange(0, self.total_frames - 1)
        self.check_ready_to_proceed()

    def check_ready_to_proceed(self):
        if self.video_loaded and self.model_loaded:
            self.proceed_button.setEnabled(True)
            self.camera_button.setEnabled(True)
        elif self.model_loaded:
            self.camera_button.setEnabled(True)

    def go_to_processing_page(self):
        self.stacked_widget.setCurrentWidget(self.second_page)
        self.label.setText("Video and model loaded, click on the video screen to start...")
        self.start_processing()

    def go_to_camera_page(self):
        self.stacked_widget.setCurrentWidget(self.third_page)
        self.start_camera()

    def start_processing(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.processed_frames = 0
        self.processed_video = []
        self.detection_results_per_frame = {}  # Store detection results for each frame
        self.update_frame_info()
        self.timer.start(30)

    def update_frame(self):
        if not self.is_seeking and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.processed_frames += 1
                results = self.model(frame)
                processed_frame = self.process_frame(frame, results)
                self.processed_video.append((processed_frame, results))  # Store with detection results
                current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

                # Print detection results
                self.print_detection_results(results)

                # Convert from OpenCV BGR format to QImage RGB format
                rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                image = QImage(rgb_frame, rgb_frame.shape[1], rgb_frame.shape[0], QImage.Format_RGB888)

                # Scale to QLabel size
                scaled_image = image.scaled(self.video_label.size(), Qt.KeepAspectRatio)
                self.video_label.setPixmap(QPixmap.fromImage(scaled_image))

                self.seek_bar.setValue(current_frame)
                self.update_frame_info()
            else:
                print("End of video.")
                self.cap.release()
                self.timer.stop()
                self.play_again_button.setEnabled(True)  # Enable the play again button when processing is complete
                self.download_button.setEnabled(True)  # Enable the download button when processing is complete

    def update_frame_info(self):
        self.frame_info_label.setText(f"Processed Frame: {self.processed_frames} / Total Frames: {self.total_frames}")

    def process_frame(self, frame, results):
        if self.model:
            processed_frame = frame.copy()  # Start with the original frame
            for result in results:
                processed_frame = result.plot()  # Plot the results on the frame
            return processed_frame
        else:
            return frame

    def print_detection_results(self, results):
        if self.model and len(results) > 0:
            detection_info = ""
            for result in results:
                if hasattr(result, 'boxes') and result.boxes is not None:
                    for box in result.boxes:
                        cls = int(box.cls[0])
                        conf = box.conf[0]
                        xyxy = box.xyxy[0]
                        detection_info += f"Detected {self.model.names[cls]} with confidence {conf:.2f} at {xyxy}\n"
            self.detection_results_text.setText(detection_info)
            self.adjust_detection_text_size()

    def adjust_detection_text_size(self):
        self.detection_results_text.adjustSize()
        self.detection_results_text.setFixedHeight(self.detection_results_text.sizeHint().height())

    def seek_video(self, frame_number):
        if self.is_seeking and self.processed_video:
            if 0 <= frame_number < len(self.processed_video):
                frame, results = self.processed_video[frame_number]
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = QImage(rgb_frame, rgb_frame.shape[1], rgb_frame.shape[0], QImage.Format_RGB888)

                # Scale to QLabel size
                scaled_image = image.scaled(self.video_label.size(), Qt.KeepAspectRatio)
                self.video_label.setPixmap(QPixmap.fromImage(scaled_image))
                self.update_frame_info()

                # Show detection results for the relevant frame
                self.print_detection_results(results)

    def start_seeking(self):
        self.is_seeking = True
        self.timer.stop()

    def end_seeking(self):
        self.is_seeking = False
        self.timer.start(30)

    def play_processed_video(self):
        self.timer.stop()
        self.playback_index = 0
        self.playback_timer.start(30)
        self.play_again_button.setEnabled(False)  # Hide the
        self.play_again_button.setEnabled(False)  # Hide the play again button when replaying
        self.detection_results_text.setText("Playing...")  # Show "Playing..." message

    def playback_frame_index(self):
        if self.playback_index < len(self.processed_video):
            frame, results = self.processed_video[self.playback_index]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = QImage(rgb_frame, rgb_frame.shape[1], rgb_frame.shape[0], QImage.Format_RGB888)

            # Scale to QLabel size
            scaled_image = image.scaled(self.video_label.size(), Qt.KeepAspectRatio)
            self.video_label.setPixmap(QPixmap.fromImage(scaled_image))

            # Show detection results for the relevant frame
            self.print_detection_results(results)

            self.playback_index += 1
        else:
            self.playback_timer.stop()
            self.play_again_button.setEnabled(True)  # Enable the play again button after replay is complete

    def download_processed_video(self):
        save_path = os.path.join("C:\\Users\\Taha\\Downloads", "processed_video.avi")
        height, width, _ = self.processed_video[0][0].shape
        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'XVID'), self.fps, (width, height))

        for frame, _ in self.processed_video:
            out.write(frame)

        out.release()
        QMessageBox.information(self, "Download Complete", f"Processed video successfully saved to: {save_path}")

    def back_to_loading(self):
        self.timer.stop()
        self.playback_timer.stop()
        if self.cap:
            self.cap.release()
        self.cap = None
        self.video_loaded = False
        self.total_frames = 0
        self.processed_frames = 0
        self.processed_video = []
        self.video_path_label.setText("Selected Video: None")
        self.proceed_button.setEnabled(False)
        self.play_again_button.setEnabled(False)
        self.download_button.setEnabled(False)
        self.detection_results_text.clear()
        self.stacked_widget.setCurrentWidget(self.first_page)

    def back_to_loading_from_camera(self):
        self.camera_timer.stop()
        if self.cap:
            self.cap.release()
        self.cap = None
        self.stacked_widget.setCurrentWidget(self.first_page)

    def start_camera(self):
        self.cap = cv2.VideoCapture(1)  # Kamera indeksi burada 0 olarak değiştirildi
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Camera Error", "Unable to open camera.")
            return
        self.camera_start_button.setEnabled(False)
        self.camera_stop_button.setEnabled(True)
        self.start_time = time.time()  # FPS hesaplaması için başlangıç zamanını belirleyin
        self.frame_counter = 0  # Frame sayacını sıfırlayın
        self.camera_timer.start(30)

    def update_camera_frame(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.frame_counter += 1
                elapsed_time = time.time() - self.start_time
                if elapsed_time > 0:
                    fps = self.frame_counter / elapsed_time
                else:
                    fps = 0.0

                results = self.model(frame)
                processed_frame = self.process_frame(frame, results)

                # FPS metnini çerçeveye ekle
                cv2.putText(processed_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # Convert from OpenCV BGR format to QImage RGB format
                rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                image = QImage(rgb_frame, rgb_frame.shape[1], rgb_frame.shape[0], QImage.Format_RGB888)

                # Scale to QLabel size
                scaled_image = image.scaled(self.camera_view.size(), Qt.KeepAspectRatio)
                self.camera_view.setPixmap(QPixmap.fromImage(scaled_image))

                # Show detection results
                self.print_camera_detection_results(results)

    def stop_camera(self):
        self.camera_timer.stop()
        if self.cap:
            self.cap.release()
        self.camera_start_button.setEnabled(True)
        self.camera_stop_button.setEnabled(False)

    def print_camera_detection_results(self, results):
        if self.model and len(results) > 0:
            detection_info = ""
            for result in results:
                if hasattr(result, 'boxes') and result.boxes is not None:
                    for box in result.boxes:
                        cls = int(box.cls[0])
                        conf = box.conf[0]
                        xyxy = box.xyxy[0]
                        detection_info += f"Detected {self.model.names[cls]} with confidence {conf:.2f} at {xyxy}\n"
            self.detection_results_camera.setText(detection_info)
            self.adjust_camera_detection_text_size()

    def adjust_camera_detection_text_size(self):
        self.detection_results_camera.adjustSize()
        self.detection_results_camera.setFixedHeight(self.detection_results_camera.sizeHint().height())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = VideoProcessingApp()
    ex.show()
    sys.exit(app.exec_())
