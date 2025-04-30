import cv2
import numpy as np
import mediapipe as mp
import threading
import time
import configparser
import logging
import signal
import argparse
import sounddevice as sd

from detectors.eye_detector import calculate_ear
from detectors.mouth_detector import calculate_mar
from detectors.advanced_lip_sync import AdvancedLipSyncDetector
from utils.calculations import fps_calculation
from utils.shared_variables import SharedVariables

# Global exit flag for graceful termination.
exit_event = threading.Event()

def signal_handler(sig, frame):
    logging.info("Signal received. Exiting gracefully...")
    exit_event.set()

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Real-Time Facial Tracker for Avatar Animation")
    parser.add_argument('--config', type=str, default='config.ini', help='Path to configuration file')
    return parser.parse_args()

def capture_audio(duration=0.5, sample_rate=16000):
    """
    Capture a short audio segment from the default input device.
    Returns the audio as a one-dimensional NumPy array.
    """
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    return audio_data.flatten()

class FrameProcessor:
    def __init__(self, config, shared_vars, data_lock):
        self.config = config
        self.shared_vars = shared_vars
        self.data_lock = data_lock

        # Retrieve thresholds & other settings from config.
        self.ear_threshold = config.getfloat('Thresholds', 'EAR_THRESHOLD', fallback=0.22)
        self.mar_threshold = config.getfloat('Thresholds', 'MAR_THRESHOLD', fallback=0.5)
        self.emotion_analysis_interval = config.getint('Advanced', 'EMOTION_ANALYSIS_INTERVAL', fallback=15)
        self.run_emotion = config.getboolean('Advanced', 'RUN_EMOTION_ANALYSIS', fallback=True)

        # Initialize MediaPipe Face Mesh.
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Check if advanced lip sync is enabled.
        self.use_deep_lip_sync = config.getboolean('Advanced', 'USE_DEEP_LIP_SYNC', fallback=False)
        if self.use_deep_lip_sync:
            sample_rate = config.getint('Advanced', 'AUDIO_SAMPLE_RATE', fallback=16000)
            # Read model file path from config; default if not provided.
            model_path = config.get('Advanced', 'MODEL_PATH', fallback="models/Wav2Lip-SD-NOGAN.pt")
            self.advanced_lip_sync = AdvancedLipSyncDetector(model_path=model_path, sample_rate=sample_rate)

        self.frame_count = 0
        self.start_time = time.time()

    def process_landmarks(self, face_landmarks, width, height):
        # Compute eye aspect ratio for left and right eyes.
        raw_ear_left = calculate_ear(
            [face_landmarks.landmark[i] for i in [33, 160, 158, 133, 153, 144]],
            width, height
        )
        raw_ear_right = calculate_ear(
            [face_landmarks.landmark[i] for i in [362, 385, 387, 263, 373, 380]],
            width, height
        )
        # Compute mouth aspect ratio.
        raw_mar = calculate_mar(
            [face_landmarks.landmark[i] for i in [61, 291, 13, 14]],
            width, height
        )
        return raw_ear_left, raw_ear_right, raw_mar

    def analyze_emotion(self, face_roi):
        # Placeholder for emotion analysis logic.
        return "Neutral"

    def render_overlay(self, frame, eye_status, mouth_status, lip_sync_val, emotion_text, fps):
        cv2.putText(frame, f"Eye Blink: {'Yes' if eye_status else 'No'}", (10, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Mouth Open: {'Yes' if mouth_status else 'No'}", (10, 280),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Lip Sync: {lip_sync_val:.2f}", (10, 310),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Emotion: {emotion_text}", (10, 340),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.2f}", (500, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return frame

    def process(self, cap):
        while cap.isOpened() and not exit_event.is_set():
            ret, frame = cap.read()
            if not ret:
                logging.warning("Failed to grab frame. Retrying...")
                continue

            frame = cv2.resize(frame, (640, 480))
            height, width = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)

            # Initialize default values.
            eye_blinked = False
            mouth_open = False
            lip_sync_value = 0.0
            emotion_text = "None"

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    ear_left, ear_right, mar = self.process_landmarks(face_landmarks, width, height)
                    avg_ear = (ear_left + ear_right) / 2.0
                    eye_blinked = avg_ear < self.ear_threshold
                    mouth_open = mar > self.mar_threshold

                    # Advanced lip sync processing.
                    if self.use_deep_lip_sync:
                        landmarks_array = np.array([[lm.x * width, lm.y * height] for lm in face_landmarks.landmark])
                        x1, y1 = np.min(landmarks_array, axis=0).astype(int)
                        x2, y2 = np.max(landmarks_array, axis=0).astype(int)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(width, x2), min(height, y2)
                        face_roi = frame[y1:y2, x1:x2]
                        try:
                            audio_sample = capture_audio(
                                duration=0.5,
                                sample_rate=self.config.getint('Advanced', 'AUDIO_SAMPLE_RATE', fallback=16000)
                            )
                            lip_sync_value = self.advanced_lip_sync.calculate_lip_sync_value(face_roi, audio_sample)
                        except Exception as e:
                            logging.error(f"Advanced lip sync error: {e}")
                            lip_sync_value = 0.0

                    # Perform emotion analysis every few frames.
                    if self.frame_count % self.emotion_analysis_interval == 0:
                        emotion_text = self.analyze_emotion(frame)

                    with self.data_lock:
                        self.shared_vars.ear_left = ear_left
                        self.shared_vars.ear_right = ear_right
                        self.shared_vars.mar = mar
                        self.shared_vars.lip_sync_value = lip_sync_value
                    # Process one face per frame.
                    break

            self.frame_count, fps = fps_calculation(self.frame_count, self.start_time)
            frame = self.render_overlay(frame, eye_blinked, mouth_open, lip_sync_value, emotion_text, fps)
            cv2.imshow('Facial Tracker', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit_event.set()
                break

        cap.release()
        cv2.destroyAllWindows()
        logging.info("Program terminated.")

def main():
    args = parse_arguments()
    config = configparser.ConfigParser()
    config.read(args.config)

    # Setup logging based on configured log level.
    log_level = config.get('Logging', 'LOG_LEVEL', fallback='INFO').upper()
    numeric_level = getattr(logging, log_level, logging.INFO)
    logging.basicConfig(level=numeric_level,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting Facial Tracker.")

    # Open video capture.
    droidcam_url = config.get('Camera', 'DROIDCAM_URL', fallback='0')
    cap = cv2.VideoCapture(droidcam_url if droidcam_url != '0' else 0)
    if not cap.isOpened():
        logging.error(f"Cannot open camera '{droidcam_url}'")
        return

    shared_vars = SharedVariables()
    data_lock = threading.Lock()

    # Start WebSocket client if enabled by configuration.
    enable_ws = config.getboolean('WebSocket', 'ENABLE_WEBSOCKET', fallback=False)
    if enable_ws:
        from websocket_client import start_websocket
        websocket_thread = threading.Thread(target=start_websocket, args=(shared_vars, data_lock), daemon=True)
        websocket_thread.start()
    else:
        logging.info("WebSocket integration is disabled via config.")

    processor = FrameProcessor(config, shared_vars, data_lock)
    processor.process(cap)

if __name__ == "__main__":
    main()