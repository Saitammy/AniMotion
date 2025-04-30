import numpy as np
import time
from typing import Tuple

def calculate_distance_coords(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """
    Calculate the Euclidean distance between two 2D coordinates.

    This function uses np.hypot for numerical stability, computing:
        sqrt((x2 - x1)² + (y2 - y1)²)

    Args:
        point1 (Tuple[float, float]): The (x, y) coordinates of the first point.
        point2 (Tuple[float, float]): The (x, y) coordinates of the second point.

    Returns:
        float: The Euclidean distance between point1 and point2.
    """
    return np.hypot(point2[0] - point1[0], point2[1] - point1[1])


def fps_calculation(frame_count: int, start_time: float) -> Tuple[int, float]:
    """
    Calculate the frames per second (FPS) of a real-time process.

    This function increments the frame count and uses the elapsed time since the start
    to compute the current FPS.

    Args:
        frame_count (int): The current frame count.
        start_time (float): The starting time (obtained via time.time()).

    Returns:
        Tuple[int, float]: The updated frame count and the computed FPS.
    """
    frame_count += 1
    elapsed_time = time.time() - start_time
    fps: float = frame_count / elapsed_time if elapsed_time > 0 else 0.0
    return frame_count, fps