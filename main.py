# main.py

import cv2
import numpy as np
import mediapipe as mp
import asyncio
import threading
import time
import configparser
import logging
from detectors.eye_detector import calculate_ear
from detectors.mouth_detector import calculate_mar
from detectors.eyebrow_detector import calculate_ebr
from detectors.lip_sync import calculate_lip_sync_value
from detectors.head_pose_estimator import get_head_pose
from websocket_client import start_websocket
from utils.calculations import fps_calculation, calculate_distance_coords

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
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Start webcam feed
try:
    if DROIDCAM_URL == '0':
        DROIDCAM_URL = 0  # Use default webcam
    cap = cv2.VideoCapture(DROIDCAM_URL)
    if not cap.isOpened():
        raise IOError(f"Cannot open camera '{DROIDCAM_URL}'")
    logger.info(f"Camera '{DROIDCAM_URL}' opened successfully.")
except Exception as e:
    logger.error(f"Error initializing camera: {e}")
    exit(1)

# FPS Calculation
frame_count = 0
start_time = time.time()

# Global variables for expression values and threading lock
from utils.shared_variables import SharedVariables
data_lock = threading.Lock()
shared_vars = SharedVariables()

# Start WebSocket connection in a separate thread
websocket_thread = threading.Thread(target=start_websocket, args=(shared_vars, data_lock), daemon=True)
websocket_thread.start()

def process_frames():
    """
    Main function to process video frames, detect facial expressions, and update shared variables.
    """
    global frame_count, start_time
    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to grab frame. Retrying...")
                continue

            # Resize frame for performance
            frame = cv2.resize(frame, (640, 480))

            # Update camera parameters based on frame size
            height, width = frame.shape[:2]
            focal_length = width
            center = (width / 2, height / 2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype="double")

            # Convert the BGR image to RGB before processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process frame for facial landmarks
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Define landmark indices
                    left_eye_indices = [33, 160, 158, 133, 153, 144]
                    right_eye_indices = [362, 385, 387, 263, 373, 380]
                    mouth_indices = [61, 291, 13, 14]
                    left_eyebrow_indices = [70, 63, 105]
                    right_eyebrow_indices = [336, 296, 334]
                    nose_tip_index = 1
                    lip_sync_indices = [13, 14, 61, 291]

                    # Extract landmarks safely
                    try:
                        left_eye_landmarks = [face_landmarks.landmark[i] for i in left_eye_indices]
                        right_eye_landmarks = [face_landmarks.landmark[i] for i in right_eye_indices]
                        mouth_landmarks = [face_landmarks.landmark[i] for i in mouth_indices]
                        left_eyebrow_landmarks = [face_landmarks.landmark[i] for i in left_eyebrow_indices]
                        right_eyebrow_landmarks = [face_landmarks.landmark[i] for i in right_eyebrow_indices]
                        nose_tip = face_landmarks.landmark[nose_tip_index]
                        lip_sync_landmarks = [face_landmarks.landmark[i] for i in lip_sync_indices]
                    except IndexError as e:
                        logger.error(f"Landmark index error: {e}")
                        continue  # Skip this frame

                    # Calculate metrics
                    ear_left = calculate_ear(left_eye_landmarks, width, height)
                    ear_right = calculate_ear(right_eye_landmarks, width, height)
                    mar = calculate_mar(mouth_landmarks, width, height)
                    ebr_left = calculate_ebr(left_eyebrow_landmarks, left_eye_landmarks, width, height)
                    ebr_right = calculate_ebr(right_eyebrow_landmarks, right_eye_landmarks, width, height)
                    lip_sync_value = calculate_lip_sync_value(lip_sync_landmarks, width, height)

                    # Head pose estimation
                    # 2D image points
                    try:
                        image_points = np.array([
                            (nose_tip.x * width, nose_tip.y * height),  # Nose tip
                            (face_landmarks.landmark[152].x * width, face_landmarks.landmark[152].y * height),  # Chin
                            (face_landmarks.landmark[263].x * width, face_landmarks.landmark[263].y * height),  # Right eye corner
                            (face_landmarks.landmark[33].x * width, face_landmarks.landmark[33].y * height),    # Left eye corner
                            (face_landmarks.landmark[287].x * width, face_landmarks.landmark[287].y * height),  # Mouth right corner
                            (face_landmarks.landmark[57].x * width, face_landmarks.landmark[57].y * height)     # Mouth left corner
                        ], dtype="double")

                        rotation_vector, translation_vector = get_head_pose(image_points, camera_matrix, dist_coeffs)
                        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                        pose_mat = cv2.hconcat((rotation_matrix, translation_vector))
                        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
                        yaw, pitch, roll = euler_angles.flatten()
                    except Exception as e:
                        logger.error(f"Head pose estimation error: {e}")
                        yaw = pitch = roll = 0.0

                    # Synchronize access to shared data
                    with data_lock:
                        shared_vars.ear_left = ear_left
                        shared_vars.ear_right = ear_right
                        shared_vars.mar = mar
                        shared_vars.ebr_left = ebr_left
                        shared_vars.ebr_right = ebr_right
                        shared_vars.lip_sync_value = lip_sync_value
                        shared_vars.yaw = yaw
                        shared_vars.pitch = pitch
                        shared_vars.roll = roll

                    # Display expressions and head pose angles
                    cv2.putText(frame, f"EAR: {ear_left:.2f}, {ear_right:.2f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(frame, f"MAR: {mar:.2f}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(frame, f"EBR: {ebr_left:.2f}, {ebr_right:.2f}", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(frame, f"Lip Sync: {lip_sync_value:.2f}", (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(frame, f"Yaw: {yaw:.2f}", (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame, f"Pitch: {pitch:.2f}", (10, 180),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame, f"Roll: {roll:.2f}", (10, 210),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            else:
                logger.debug("No face landmarks detected.")

            # Calculate FPS
            frame_count, fps = fps_calculation(frame_count, start_time)
            cv2.putText(frame, f"FPS: {fps:.2f}", (500, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Display the frame using OpenCV
            cv2.imshow('Facial Tracker', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Quit command received. Exiting...")
                break

    except Exception as e:
        logger.exception(f"An error occurred during frame processing: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Resources released, and program terminated.")

if __name__ == "__main__":
    process_frames()
