from typing import List, Tuple, Any
from utils.calculations import calculate_distance_coords

def calculate_mar(mouth_landmarks: List[Any], width: int, height: int) -> float:
    """
    Calculate the Mouth Aspect Ratio (MAR) using four selected mouth landmarks.

    The function expects that:
      - landmarks with indices [0] and [1] define the horizontal extent of the mouth.
      - landmarks with indices [2] and [3] define the vertical distance.

    The MAR is computed as:
        MAR = vertical_distance / horizontal_distance

    :param mouth_landmarks: List of exactly 4 landmarks. Each landmark must have 'x' and 'y' attributes.
    :param width: Width of the frame (in pixels).
    :param height: Height of the frame (in pixels).
    :return: The computed Mouth Aspect Ratio (MAR) as a float.
    :raises ValueError: if mouth_landmarks does not contain exactly 4 elements.
    """
    if len(mouth_landmarks) != 4:
        raise ValueError(f"Expected exactly 4 mouth landmarks, but got {len(mouth_landmarks)}.")

    # Convert normalized landmarks to pixel coordinates.
    coords: List[Tuple[int, int]] = [
        (int(point.x * width), int(point.y * height)) for point in mouth_landmarks
    ]

    # Calculate horizontal and vertical distances.
    horizontal = calculate_distance_coords(coords[0], coords[1])
    vertical = calculate_distance_coords(coords[2], coords[3])

    # Prevent division by zero.
    if horizontal == 0:
        return 0.0

    mar = vertical / horizontal
    return mar