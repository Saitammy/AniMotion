# detectors/eyebrow_detector.py

from utils.calculations import calculate_distance

def calculate_ebr(eyebrow_landmarks, eye_landmarks, width, height):
    """
    Calculate Eyebrow Raise Ratio (EBR)
    :param eyebrow_landmarks: List of eyebrow landmarks (left or right)
    :param eye_landmarks: List of eye landmarks (left or right)
    :return: Eyebrow raise ratio
    """
    # Select central points
    eyebrow_point = eyebrow_landmarks[1]
    eye_point = eye_landmarks[1]

    # Convert landmarks to coordinates
    eyebrow_coord = (int(eyebrow_point.x * width), int(eyebrow_point.y * height))
    eye_coord = (int(eye_point.x * width), int(eye_point.y * height))

    # Calculate vertical distance
    vertical_distance = calculate_distance_coords(eyebrow_coord, eye_coord)

    # Normalize by eye width
    eye_width = calculate_distance_coords(
        (int(eye_landmarks[0].x * width), int(eye_landmarks[0].y * height)),
        (int(eye_landmarks[3].x * width), int(eye_landmarks[3].y * height))
    )

    ebr = vertical_distance / eye_width if eye_width else 0

    return ebr

def calculate_distance_coords(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5
