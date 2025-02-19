import cv2
import numpy as np
import mediapipe as mp
import asyncio
import threading
import time
import configparser
import logging
import torch
from torchvision import transforms
from PIL import Image
from detectors.eye_detector import calculate_ear
from detectors.mouth_detector import calculate_mar
from detectors.eyebrow_detector import calculate_ebr
from detectors.lip_sync import calculate_lip_sync_value
from detectors.head_pose_estimator import get_head_pose
from websocket_client import start_websocket
from utils.calculations import fps_calculation
from utils.shared_variables import SharedVariables

# Load configuration
config = configparser.ConfigParser()
config.read('config.ini')

# Set up logging
log_level = config.get('Logging', 'LOG_LEVEL', fallback='INFO').upper()
numeric_level = getattr(logging, log_level, logging.INFO)
logging.basicConfig(level=numeric_level, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Retrieve configuration parameters
DROIDCAM_URL = config.get('Camera', 'DROIDCAM_URL', fallback='0')
EAR_THRESHOLD = config.getfloat('Thresholds', 'EAR_THRESHOLD', fallback=0.22)
MAR_THRESHOLD = config.getfloat('Thresholds', 'MAR_THRESHOLD', fallback=0.5)
EBR_THRESHOLD = config.getfloat('Thresholds', 'EBR_THRESHOLD', fallback=1.5)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=2, refine_landmarks=True,
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Start webcam feed
cap = cv2.VideoCapture(DROIDCAM_URL if DROIDCAM_URL != '0' else 0)
if not cap.isOpened():
    logger.error(f"Cannot open camera '{DROIDCAM_URL}'")
    exit(1)

# FPS Calculation
frame_count = 0
start_time = time.time()
frame_skip = 2  # Skip every 2nd frame to reduce delay

data_lock = threading.Lock()
shared_vars = SharedVariables()

# Start WebSocket thread
websocket_thread = threading.Thread(target=start_websocket, args=(shared_vars, data_lock), daemon=True)
websocket_thread.start()

# Rolling history for adaptive thresholds
ear_history = []
lip_history = []

def process_frames():
    global frame_count, start_time
    dist_coeffs = np.zeros((4, 1))
    
    while cap.isOpened():
        eye_blinked = "No"
        mouth_open = "No"
        lip_sync_active = "No"
        
        ret, frame = cap.read()
        
        if frame_count % frame_skip != 0:
            frame_count += 1
            continue

        if not ret:
            logger.warning("Failed to grab frame. Retrying...")
            continue

        frame = cv2.resize(frame, (640, 480))
        height, width = frame.shape[:2]
        focal_length = width
        center = (width / 2, height / 2)
        camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double")

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                ear_left = calculate_ear([face_landmarks.landmark[i] for i in [33, 160, 158, 133, 153, 144]], width, height)
                ear_right = calculate_ear([face_landmarks.landmark[i] for i in [362, 385, 387, 263, 373, 380]], width, height)
                mar = calculate_mar([face_landmarks.landmark[i] for i in [61, 291, 13, 14]], width, height)
                lip_sync_value = calculate_lip_sync_value([face_landmarks.landmark[i] for i in [13, 14, 61, 291]], width, height)
                
                # Adaptive EAR threshold
                avg_ear = (ear_left + ear_right) / 2
                ear_history.append(avg_ear)
                if len(ear_history) > 50:
                    ear_history.pop(0)
                adaptive_ear_threshold = max(min(ear_history) * 0.85, 0.18)
                print(f"Adaptive EAR Threshold: {adaptive_ear_threshold}, Current EAR: {avg_ear}")
                eye_blinked = "Yes" if ear_left < adaptive_ear_threshold and ear_right < adaptive_ear_threshold else "No"
                
                # Adaptive Lip Sync detection
                lip_history.append(lip_sync_value)
                if len(lip_history) > 30:
                    lip_history.pop(0)
                smoothed_lip_sync = sum(lip_history[-10:]) / min(len(lip_history), 10)
                print(f"Smoothed Lip Sync: {smoothed_lip_sync}, Raw Value: {lip_sync_value}")
                lip_sync_active = "Yes" if smoothed_lip_sync > 0.12 else "No"
                
                mouth_open = "Yes" if mar > MAR_THRESHOLD else "No"
        
        frame_count, fps = fps_calculation(frame_count, start_time)
        cv2.putText(frame, f"Eye Blinked: {eye_blinked}", (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Mouth Open: {mouth_open}", (10, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Lip Sync Active: {lip_sync_active}", (10, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.2f}", (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow('Facial Tracker', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Program terminated.")

if __name__ == "__main__":
    process_frames()