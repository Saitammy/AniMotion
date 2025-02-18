# detectors/eye_detector.py

from utils.calculations import calculate_distance

def calculate_ear(eye_landmarks, width, height):
    """
    Calculate Eye Aspect Ratio (EAR)
    :param eye_landmarks: List of eye landmarks
    :return: Eye aspect ratio
    """
    # Convert landmarks to coordinates
    coords = [(int(point.x * width), int(point.y * height)) for point in eye_landmarks]

    # Calculate distances
    vertical1 = calculate_distance_coords(coords[1], coords[5])
    vertical2 = calculate_distance_coords(coords[2], coords[4])
    horizontal = calculate_distance_coords(coords[0], coords[3])

    # Calculate EAR
    ear = (vertical1 + vertical2) / (2.0 * horizontal) if horizontal else 0

    return ear

def calculate_distance_coords(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5
