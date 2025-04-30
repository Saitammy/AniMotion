from typing import List, Tuple, Any
from utils.calculations import calculate_distance_coords

def calculate_lip_sync_value(lip_landmarks: List[Any], width: int, height: int) -> float:
    """
    Calculate the Lip Sync Value using selected lip landmarks.

    This function is designed to measure the intensity of lip movements, which can then be mapped to
    a lip synchronization parameter for the animated model. It uses at least four lip landmarks:
      - landmarks[0] and landmarks[1] are assumed to represent the outer horizontal lip corners.
      - landmarks[2] and landmarks[3] are assumed to represent the inner vertical boundaries of the lips.
    
    The lip sync value is computed as the ratio between the vertical distance (inner lip gap)
    and the horizontal distance (outer lip width). A higher ratio can indicate a more pronounced
    lip movement, e.g., when speaking.

    :param lip_landmarks: List of at least 4 landmarks; each landmark should have 'x' and 'y' attributes,
                          with normalized values (0 to 1).
    :param width: Width of the frame (in pixels).
    :param height: Height of the frame (in pixels).
    :return: A float representing the lip sync value.
    :raises ValueError: If fewer than 4 landmarks are provided.
    """
    if len(lip_landmarks) < 4:
        raise ValueError(f"Expected at least 4 lip landmarks, but got {len(lip_landmarks)}.")

    # Convert normalized coordinates to pixel positions.
    coords: List[Tuple[int, int]] = [
        (int(point.x * width), int(point.y * height)) for point in lip_landmarks
    ]
    
    # Calculate the outer horizontal distance between the lip corners (landmarks[0] and landmarks[1]).
    outer_distance = calculate_distance_coords(coords[0], coords[1])
    
    # Calculate the inner vertical distance between the upper and lower lips (landmarks[2] and landmarks[3]).
    inner_distance = calculate_distance_coords(coords[2], coords[3])
    
    # Guard against division by zero.
    if outer_distance == 0:
        return 0.0

    # Compute the lip sync value as the ratio of vertical motion to the horizontal lip span.
    lip_sync_value = inner_distance / outer_distance
    return lip_sync_value