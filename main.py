import cv2
import numpy as np
import mediapipe as mp
import asyncio
import threading
import time
from detectors.eye_detector import calculate_ear
from detectors.mouth_detector import calculate_mar
from detectors.eyebrow_detector import calculate_ebr
from detectors.lip_sync import calculate_lip_sync_value
from websocket_client import start_websocket
from utils.calculations import calculate_distance, fps_calculation
from detectors.head_pose_estimator import get_head_pose

# Initialize MediaPipe Face Mesh with optimized parameters
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,  # Use refined landmarks for better accuracy
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Start webcam feed
DROIDCAM_URL = "http://192.168.102.71:4747/video"  # Update with your camera's URL or use 0 for default webcam
cap = cv2.VideoCapture(DROIDCAM_URL)

# Thresholds (Adjust these values based on calibration and testing)
EAR_THRESHOLD = 0.22   # Eye Aspect Ratio threshold for blink detection
MAR_THRESHOLD = 0.5    # Mouth Aspect Ratio threshold for mouth open detection
EBR_THRESHOLD = 1.5    # Eyebrow Raise Ratio threshold

# FPS Calculation
frame_count = 0
start_time = time.time()

# Global variables for expression values
ear_left = None
ear_right = None
mar = None
ebr_left = None
ebr_right = None
lip_sync_value = None

# Start WebSocket connection in a separate thread
threading.Thread(target=start_websocket, daemon=True).start()

def process_frames():
    global ear_left, ear_right, mar, ebr_left, ebr_right, lip_sync_value, frame_count, start_time

    # Camera parameters for head pose estimation
    focal_length = 1  # Will be updated based on frame dimensions
    center = None     # Will be updated based on frame dimensions
    camera_matrix = None
    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Check camera connection.")
            break

        # Resize frame to improve performance
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
                mouth_indices = [78, 308, 13, 14]
                left_eyebrow_indices = [70, 63, 105]    # Adjusted indices for left eyebrow
                right_eyebrow_indices = [336, 296, 334] # Adjusted indices for right eyebrow
                nose_tip_index = 1                      # Index for nose tip
                lip_sync_indices = [13, 14, 78, 308]    # Indices for lip sync landmarks

                # Get landmarks
                left_eye_landmarks = [face_landmarks.landmark[i] for i in left_eye_indices]
                right_eye_landmarks = [face_landmarks.landmark[i] for i in right_eye_indices]
                mouth_landmarks = [face_landmarks.landmark[i] for i in mouth_indices]
                left_eyebrow_landmarks = [face_landmarks.landmark[i] for i in left_eyebrow_indices]
                right_eyebrow_landmarks = [face_landmarks.landmark[i] for i in right_eyebrow_indices]
                nose_tip = face_landmarks.landmark[nose_tip_index]
                lip_sync_landmarks = [face_landmarks.landmark[i] for i in lip_sync_indices]

                # Calculate metrics
                ear_left = calculate_ear(left_eye_landmarks, width, height)
                ear_right = calculate_ear(right_eye_landmarks, width, height)
                mar = calculate_mar(mouth_landmarks, width, height)
                ebr_left = calculate_ebr(left_eyebrow_landmarks, left_eye_landmarks, width, height)
                ebr_right = calculate_ebr(right_eyebrow_landmarks, right_eye_landmarks, width, height)
                lip_sync_value = calculate_lip_sync_value(lip_sync_landmarks, width, height)

                # Head pose estimation
                # 2D image points
                image_points = np.array([
                    (nose_tip.x * width, nose_tip.y * height),                                  # Nose tip
                    (face_landmarks.landmark[152].x * width, face_landmarks.landmark[152].y * height),  # Chin
                    (face_landmarks.landmark[263].x * width, face_landmarks.landmark[263].y * height),  # Right eye corner
                    (face_landmarks.landmark[33].x * width, face_landmarks.landmark[33].y * height),    # Left eye corner
                    (face_landmarks.landmark[287].x * width, face_landmarks.landmark[287].y * height),  # Mouth right corner
                    (face_landmarks.landmark[57].x * width, face_landmarks.landmark[57].y * height)     # Mouth left corner
                ], dtype="double")

                # Solve PnP to get rotation and translation vectors
                rotation_vector, translation_vector = get_head_pose(image_points, camera_matrix, dist_coeffs)

                # Convert rotation vector to Euler angles
                rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                pose_mat = cv2.hconcat((rotation_matrix, translation_vector))
                _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
                yaw, pitch, roll = euler_angles.flatten()

                # Map the calculated metrics to control your avatar
                # Here you can send these values through WebSocket or to any avatar animation interface

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

                # Optional: Draw facial landmarks for visualization
                # for landmark in face_landmarks.landmark:
                #     x = int(landmark.x * width)
                #     y = int(landmark.y * height)
                #     cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        # Calculate FPS
        frame_count, fps = fps_calculation(frame_count, start_time)
        cv2.putText(frame, f"FPS: {fps:.2f}", (500, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display the frame using OpenCV
        cv2.imshow('Facial Tracker', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Resources released, and program terminated.")

# Start Frame Processing
if __name__ == "__main__":
    process_frames()
