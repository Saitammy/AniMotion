import logging
from utils.calculations import calculate_distance
import numpy as np

class EyebrowDetector:
    def __init__(self):
        self.indices = {
            'left': [70, 63, 105, 66, 107],
            'right': [336, 296, 334, 293, 300]
        }
        # Initialize logging
        logging.basicConfig(level=logging.INFO)

    def detect_eyebrows(self, landmarks, frame_size):
        if not landmarks or not frame_size:
            logging.error("Invalid landmarks or frame size")
            return {}

        results = {}
        for side in ['left', 'right']:
            brow_points = [landmarks[i] for i in self.indices[side]]
            eye_points = [landmarks[i] for i in ([33, 133] if side == 'left' else [362, 263])]

            distances = [calculate_distance(brow, eye_points[0], *frame_size) for brow in brow_points]
            avg_distance = sum(distances) / len(distances)
            results[f'eyebrow_{side}_raise'] = np.interp(avg_distance, [15, 25], [0.0, 1.0])

            logging.info(f"{side.capitalize()} Eyebrow Raise: {results[f'eyebrow_{side}_raise']}")

        return results

    def visualize_eyebrows(self, frame, landmarks, results):
        import cv2
        for side in ['left', 'right']:
            color = (0, 255, 0) if results[f'eyebrow_{side}_raise'] > 0.5 else (0, 0, 255)
            points = [landmarks[i] for i in self.indices[side]]
            for point in points:
                cv2.circle(frame, (int(point.x * frame.shape[1]), int(point.y * frame.shape[0])), 2, color, -1)
        return frame
