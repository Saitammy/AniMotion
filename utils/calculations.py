import numpy as np
import time

def calculate_distance(landmark1, landmark2, width, height):
    x1, y1 = int(landmark1.x * width), int(landmark1.y * height)
    x2, y2 = int(landmark2.x * width), int(landmark2.y * height)
    return np.linalg.norm([x2 - x1, y2 - y1])

def fps_calculation(frame_count, start_time):
    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    return frame_count, fps
