import logging
from utils.calculations import calculate_distance

class LipSyncDetector:
    def __init__(self):
        self.vowel_thresholds = {
            'Aa': 0.7,
            'Ee': 0.5,
            'Oh': 0.6,
            'Uh': 0.4
        }
        logging.basicConfig(level=logging.INFO)

    def detect_vowels(self, mouth_points, frame_size):
        if not mouth_points or not frame_size:
            logging.error("Invalid mouth points or frame size")
            return {}

        vertical = calculate_distance(mouth_points[2], mouth_points[3], *frame_size)
        horizontal = calculate_distance(mouth_points[0], mouth_points[1], *frame_size)
        width = calculate_distance(mouth_points[4], mouth_points[5], *frame_size)
        
        ratios = {
            'Aa': vertical / horizontal,
            'Ee': width / horizontal,
            'Oh': (vertical + width) / (horizontal * 2),
            'Uh': vertical / (horizontal * 1.5)
        }
        
        results = {k: 1.0 if v > self.vowel_thresholds[k] else 0.0 for k, v in ratios.items()}
        
        for vowel, value in results.items():
            logging.info(f"Vowel {vowel}: {'Detected' if value else 'Not Detected'}")

        return results
