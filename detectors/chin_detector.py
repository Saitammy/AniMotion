import logging
from utils.calculations import calculate_distance

class ChinDetector:
    def __init__(self):
        self.chin_indices = [152, 176, 150, 378, 385, 400]
        # Initialize logging
        logging.basicConfig(level=logging.INFO)

    def detect_chin(self, landmarks, frame_size):
        if not landmarks or not frame_size:
            logging.error("Invalid landmarks or frame size")
            return {}

        points = [landmarks[i] for i in self.chin_indices]
        
        # Chin height and width
        height = calculate_distance(points[0], points[1], *frame_size)
        width = calculate_distance(points[2], points[3], *frame_size)

        results = {
            'chin_height': height,
            'chin_width': width
        }

        logging.info(f"Chin Height: {results['chin_height']}, Chin Width: {results['chin_width']}")

        return results

    def visualize_chin(self, frame, landmarks, results):
        import cv2
        color = (0, 255, 0) if results['chin_height'] > 20 else (0, 0, 255)
        points = [landmarks[i] for i in self.chin_indices]
        for point in points:
            cv2.circle(frame, (int(point.x * frame.shape[1]), int(point.y * frame.shape[0])), 2, color, -1)
        return frame
