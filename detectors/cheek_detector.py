import logging
from utils.calculations import calculate_distance

class CheekDetector:
    def __init__(self):
        self.cheek_indices = {
            'left': [226, 229, 130],
            'right': [446, 449, 359]
        }
        # Initialize logging
        logging.basicConfig(level=logging.INFO)

    def detect_cheeks(self, landmarks, frame_size):
        if not landmarks or not frame_size:
            logging.error("Invalid landmarks or frame size")
            return {}

        results = {}
        for side in ['left', 'right']:
            points = [landmarks[i] for i in self.cheek_indices[side]]
            distances = [calculate_distance(points[i], points[j], *frame_size) for i in range(len(points)) for j in range(i + 1, len(points))]
            avg_distance = sum(distances) / len(distances)
            results[f'cheek_{side}_raise'] = avg_distance

            logging.info(f"{side.capitalize()} Cheek Raise: {results[f'cheek_{side}_raise']}")

        return results

    def visualize_cheeks(self, frame, landmarks, results):
        import cv2
        for side in ['left', 'right']:
            color = (0, 255, 0) if results[f'cheek_{side}_raise'] > 20 else (0, 0, 255)
            points = [landmarks[i] for i in self.cheek_indices[side]]
            for point in points:
                cv2.circle(frame, (int(point.x * frame.shape[1]), int(point.y * frame.shape[0])), 2, color, -1)
        return frame
