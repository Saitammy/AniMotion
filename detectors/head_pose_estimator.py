# detectors/head_pose_estimator.py

import cv2
import numpy as np

# 3D model points of facial landmarks
model_points = np.array([
    (0.0, 0.0, 0.0),          # Nose tip
    (0.0, -330.0, -65.0),     # Chin
    (-225.0, 170.0, -135.0),  # Left eye corner
    (225.0, 170.0, -135.0),   # Right eye corner
    (-150.0, -150.0, -125.0), # Left mouth corner
    (150.0, -150.0, -125.0)   # Right mouth corner
])

def get_head_pose(image_points, camera_matrix, dist_coeffs):
    """
    Calculate head pose using solvePnP.

    :param image_points: 2D image points from facial landmarks
    :param camera_matrix: Camera matrix computed from frame dimensions
    :param dist_coeffs: Distortion coefficients (assumed zero here)
    :return: rotation_vector, translation_vector
    """
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    return rotation_vector, translation_vector
