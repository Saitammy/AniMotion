import cv2
import mediapipe as mp
import time
from detectors.eye_detector import calculate_ear
from detectors.mouth_detector import calculate_mar

def calibrate_camera(num_frames=100):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    ear_values = []
    mar_values = []
    count = 0
    print("Starting calibration: Please maintain a neutral face pose...")
    start_time = time.time()
    while count < num_frames:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break
        frame = cv2.resize(frame, (640, 480))
        height, width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            ear_left = calculate_ear([face_landmarks.landmark[i] for i in [33,160,158,133,153,144]], width, height)
            ear_right = calculate_ear([face_landmarks.landmark[i] for i in [362,385,387,263,373,380]], width, height)
            mar = calculate_mar([face_landmarks.landmark[i] for i in [61,291,13,14]], width, height)
            avg_ear = (ear_left + ear_right) / 2.0
            ear_values.append(avg_ear)
            mar_values.append(mar)
            count += 1
            cv2.putText(frame, f'Calibrating: {count}/{num_frames}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.imshow("Calibration", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if ear_values and mar_values:
        avg_ear_final = sum(ear_values) / len(ear_values)
        avg_mar_final = sum(mar_values) / len(mar_values)
        # Suggest thresholds: You may experiment with the margin factors.
        suggested_ear_threshold = avg_ear_final * 0.9
        suggested_mar_threshold = avg_mar_final * 1.1
        print("Calibration complete!")
        print(f"Average EAR: {avg_ear_final:.3f} -> Suggested EAR threshold: {suggested_ear_threshold:.3f}")
        print(f"Average MAR: {avg_mar_final:.3f} -> Suggested MAR threshold: {suggested_mar_threshold:.3f}")
    else:
        print("Calibration failed: No face detected.")

if __name__ == "__main__":
    calibrate_camera()