import cv2
import numpy as np
import mediapipe as mp
import asyncio
import threading
import time
from detectors.eye_detector import calculate_ear
from detectors.mouth_detector import calculate_mar
from websocket_client import start_websocket
from utils.calculations import calculate_distance, fps_calculation

# Initialize MediaPipe Face Mesh with optimized parameters
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,  # Set to False if not using refined landmarks
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Start webcam feed
DROIDCAM_URL = "http://192.168.102.71:4747/video"
cap = cv2.VideoCapture(DROIDCAM_URL)

# Thresholds
EAR_THRESHOLD = 0.22  # Blink detection
MAR_THRESHOLD = 0.4   # Mouth open detection

# FPS Calculation
frame_count = 0
start_time = time.time()

# Start WebSocket connection in a separate thread
threading.Thread(target=start_websocket, daemon=True).start()

def process_frames():
    global ear_left, ear_right, mar, frame_count, start_time

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Check DroidCam connection.")
            break

        # Resize frame to improve performance if needed
        frame = cv2.resize(frame, (640, 480))

        # Convert the BGR image to RGB before processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame for facial landmarks
        results = face_mesh.process(rgb_frame)
        height, width = frame.shape[:2]

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Eye indices (using the same indices as your original code)
                left_eye_indices = [33, 160, 158, 133, 153, 144]
                right_eye_indices = [362, 385, 386, 263, 373, 380]
                mouth_indices = [78, 308, 13, 14]

                left_eye_landmarks = [face_landmarks.landmark[i] for i in left_eye_indices]
                right_eye_landmarks = [face_landmarks.landmark[i] for i in right_eye_indices]
                mouth_landmarks = [face_landmarks.landmark[i] for i in mouth_indices]

                ear_left = calculate_ear(left_eye_landmarks, width, height)
                ear_right = calculate_ear(right_eye_landmarks, width, height)
                mar = calculate_mar(mouth_landmarks, width, height)

                # Blink detection
                if ear_left < EAR_THRESHOLD and ear_right < EAR_THRESHOLD:
                    cv2.putText(frame, "Blink Detected!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # Mouth open detection
                if mar > MAR_THRESHOLD:
                    cv2.putText(frame, "Mouth Open!", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Display EAR and MAR values
                cv2.putText(frame, f"EAR: {ear_left:.2f}, {ear_right:.2f}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"MAR: {mar:.2f}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Calculate FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (500, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display the frame using OpenCV
        cv2.imshow('Facial Landmarks', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Resources released, and program terminated.")

# Start Frame Processing
if __name__ == "__main__":
    process_frames()
