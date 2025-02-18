# utils/calculations.py

import numpy as np
import time

def calculate_distance(landmark1, landmark2, width, height):
    """
    Calculate Euclidean distance between two landmarks
    """
    x1, y1 = int(landmark1.x * width), int(landmark1.y * height)
    x2, y2 = int(landmark2.x * width), int(landmark2.y * height)
    return np.hypot(x2 - x1, y2 - y1)

def fps_calculation(frame_count, start_time):
    """
    Calculate the frames per second (FPS)
    """
    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    return frame_count, fps
