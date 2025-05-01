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
import tkinter as tk  # For the diagnostics dashboard

# Attempt to import DeepFace for emotion recognition.
try:
    from deepface import DeepFace
    deepface_available = True
except ImportError:
    logging.warning("DeepFace is not installed. Emotion recognition will return 'Neutral'.")
    deepface_available = False

from detectors.eye_detector import calculate_ear
from detectors.mouth_detector import calculate_mar
from detectors.advanced_lip_sync import AdvancedLipSyncDetector
from utils.calculations import fps_calculation
from utils.shared_variables import SharedVariables

# Global exit event for graceful termination.
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

def start_dashboard(shared_vars, data_lock):
    """
    Launch a diagnostics dashboard using Tkinter.
    It displays FPS and detailed metrics per detected face including:
    EAR values, MAR value, blink status (Yes/No), mouth open status (Yes/No),
    lip sync value and detected emotion.
    """
    root = tk.Tk()
    root.title("Diagnostics Dashboard")
    label = tk.Label(root, text="Starting...", font=("Helvetica", 14))
    label.pack(padx=20, pady=20)

    def update():
        with data_lock:
            fps = shared_vars.fps if shared_vars.fps is not None else 0.0
            faces = shared_vars.faces
        info = f"FPS: {fps:.2f}\nFaces Detected: {len(faces)}\n"
        for idx, face in enumerate(faces):
            ear_left = face.get("ear_left", "N/A")
            ear_right = face.get("ear_right", "N/A")
            mar = face.get("mar", "N/A")
            blink = face.get("eye_blinked", False)
            mouth_open = face.get("mouth_open", False)
            lip_sync = face.get("lip_sync_value", "N/A")
            emotion = face.get("emotion", "N/A")
            info += f"\nFace {idx+1}:\n"
            info += f"  EAR Left : {ear_left if isinstance(ear_left, str) else f'{ear_left:.2f}'}\n"
            info += f"  EAR Right: {ear_right if isinstance(ear_right, str) else f'{ear_right:.2f}'}\n"
            info += f"  Blink    : {'Yes' if blink else 'No'}\n"
            info += f"  MAR      : {mar if isinstance(mar, str) else f'{mar:.2f}'}\n"
            info += f"  Mouth Open: {'Yes' if mouth_open else 'No'}\n"
            info += f"  Lip Sync : {lip_sync if isinstance(lip_sync, str) else f'{lip_sync:.2f}'}\n"
            info += f"  Emotion  : {emotion}\n"
        label.config(text=info)
        root.after(1000, update)

    update()
    root.mainloop()

class FrameProcessor:
    def __init__(self, config, shared_vars, data_lock):
        self.config = config
        self.shared_vars = shared_vars
        self.data_lock = data_lock

        # Retrieve thresholds & settings from config.
        self.ear_threshold = config.getfloat('Thresholds', 'EAR_THRESHOLD', fallback=0.22)
        self.mar_threshold = config.getfloat('Thresholds', 'MAR_THRESHOLD', fallback=0.5)
        self.emotion_analysis_interval = config.getint('Advanced', 'EMOTION_ANALYSIS_INTERVAL', fallback=15)
        self.use_emotion_recognition = config.getboolean('Emotion', 'USE_EMOTION_RECOGNITION', fallback=True)
        self.run_emotion = config.getboolean('Advanced', 'RUN_EMOTION_ANALYSIS', fallback=True)
        self.lip_sync_skip = config.getint('Advanced', 'LIP_SYNC_SKIP_FRAMES', fallback=3)
        self.previous_lip_sync_value = 0.0

        # Initialize MediaPipe Face Mesh.
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=5,  # Allow multiple faces.
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Advanced lip sync module.
        self.use_deep_lip_sync = config.getboolean('Advanced', 'USE_DEEP_LIP_SYNC', fallback=False)
        if self.use_deep_lip_sync:
            sample_rate = config.getint('Advanced', 'AUDIO_SAMPLE_RATE', fallback=16000)
            model_path = config.get('Advanced', 'MODEL_PATH', fallback="models/Wav2Lip-SD-NOGAN.pt")
            self.advanced_lip_sync = AdvancedLipSyncDetector(model_path=model_path, sample_rate=sample_rate)

        self.frame_count = 0
        self.start_time = time.time()

    def analyze_emotion(self, face_roi):
        """
        Analyze the face ROI and return the dominant emotion.
        If DeepFace returns a list, use the first element.
        """
        if self.use_emotion_recognition and deepface_available:
            try:
                # Specify a detector backend, e.g., "opencv".
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
                if isinstance(result, list):
                    result = result[0]
                return result.get("dominant_emotion", "Neutral")
            except Exception as e:
                logging.error("Emotion analysis error: " + str(e))
                return "Neutral"
        else:
            return "Neutral"

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
            
            face_data = []  # List to hold metrics for each detected face.

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Compute EAR and MAR.
                    ear_left = calculate_ear([face_landmarks.landmark[i] for i in [33, 160, 158, 133, 153, 144]], width, height)
                    ear_right = calculate_ear([face_landmarks.landmark[i] for i in [362, 385, 387, 263, 373, 380]], width, height)
                    mar = calculate_mar([face_landmarks.landmark[i] for i in [61, 291, 13, 14]], width, height)
                    avg_ear = (ear_left + ear_right) / 2.0
                    # Determine blink based on EAR threshold.
                    eye_blinked = avg_ear < self.ear_threshold
                    # Determine mouth open based on MAR threshold.
                    mouth_open = mar > self.mar_threshold

                    # Default advanced metrics.
                    lip_sync_value = self.previous_lip_sync_value
                    emotion_text = "Neutral"

                    # Run advanced lip sync (and emotion analysis) on every Nth frame.
                    if self.use_deep_lip_sync and (self.frame_count % self.lip_sync_skip == 0):
                        landmarks_array = np.array([[lm.x * width, lm.y * height] for lm in face_landmarks.landmark])
                        x1, y1 = np.min(landmarks_array, axis=0).astype(int)
                        x2, y2 = np.max(landmarks_array, axis=0).astype(int)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(width, x2), min(height, y2)
                        face_roi = frame[y1:y2, x1:x2]
                        try:
                            audio_sample = capture_audio(duration=0.5, sample_rate=self.config.getint('Advanced', 'AUDIO_SAMPLE_RATE', fallback=16000))
                            lip_sync_value = self.advanced_lip_sync.calculate_lip_sync_value(face_roi, audio_sample)
                            self.previous_lip_sync_value = lip_sync_value
                        except Exception as e:
                            logging.error(f"Advanced lip sync error: {e}")
                            lip_sync_value = 0.0

                        emotion_text = self.analyze_emotion(face_roi)

                    # Compute face bounding box.
                    landmarks_array = np.array([[lm.x * width, lm.y * height] for lm in face_landmarks.landmark])
                    x1, y1 = np.min(landmarks_array, axis=0).astype(int)
                    x2, y2 = np.max(landmarks_array, axis=0).astype(int)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)
                    
                    # Append this face's metrics.
                    face_data.append({
                        "ear_left": ear_left,
                        "ear_right": ear_right,
                        "mar": mar,
                        "eye_blinked": eye_blinked,
                        "mouth_open": mouth_open,
                        "lip_sync_value": lip_sync_value,
                        "emotion": emotion_text,
                        "bounding_box": (x1, y1, x2, y2)
                    })

                    # Draw overlay for this face.
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"LS:{lip_sync_value:.2f}", (x1, y1 - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                    cv2.putText(frame, emotion_text, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

            # Update shared variables.
            with self.data_lock:
                self.shared_vars.faces = face_data
            self.frame_count, fps = fps_calculation(self.frame_count, self.start_time)
            with self.data_lock:
                self.shared_vars.fps = fps

            # Overlay FPS on the frame.
            cv2.putText(frame, f"FPS: {fps:.2f}", (500, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
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

    # Setup logging.
    log_level = config.get('Logging', 'LOG_LEVEL', fallback='INFO').upper()
    numeric_level = getattr(logging, log_level, logging.INFO)
    logging.basicConfig(level=numeric_level, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting Facial Tracker.")

    # Open video capture.
    droidcam_url = config.get('Camera', 'DROIDCAM_URL', fallback='0')
    cap = cv2.VideoCapture(droidcam_url if droidcam_url != '0' else 0)
    if not cap.isOpened():
        logging.error(f"Cannot open camera '{droidcam_url}'")
        return

    shared_vars = SharedVariables()
    shared_vars.faces = []
    shared_vars.fps = 0.0

    data_lock = threading.Lock()

    # Optionally, start the WebSocket client if enabled.
    enable_ws = config.getboolean('WebSocket', 'ENABLE_WEBSOCKET', fallback=False)
    if enable_ws:
        from websocket_client import start_websocket
        websocket_thread = threading.Thread(target=start_websocket, args=(shared_vars, data_lock), daemon=True)
        websocket_thread.start()
    else:
        logging.info("WebSocket integration is disabled via config.")

    # Optionally, start the diagnostics dashboard.
    enable_dashboard = config.getboolean('Dashboard', 'ENABLE_DASHBOARD', fallback=True)
    if enable_dashboard:
        dashboard_thread = threading.Thread(target=start_dashboard, args=(shared_vars, data_lock), daemon=True)
        dashboard_thread.start()
    else:
        logging.info("Diagnostics dashboard is disabled via config.")

    processor = FrameProcessor(config, shared_vars, data_lock)
    processor.process(cap)

if __name__ == "__main__":
    main()