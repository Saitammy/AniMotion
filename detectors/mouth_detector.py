from utils.calculations import calculate_distance

def calculate_mar(mouth_landmarks, width, height):
    vertical = calculate_distance(mouth_landmarks[2], mouth_landmarks[3], width, height)
    horizontal = calculate_distance(mouth_landmarks[0], mouth_landmarks[1], width, height)
    return vertical / horizontal if horizontal else 0
