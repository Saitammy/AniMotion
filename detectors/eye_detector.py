from utils.calculations import calculate_distance

def calculate_ear(eye_landmarks, width, height):
    vertical1 = calculate_distance(eye_landmarks[1], eye_landmarks[5], width, height)
    vertical2 = calculate_distance(eye_landmarks[2], eye_landmarks[4], width, height)
    horizontal = calculate_distance(eye_landmarks[0], eye_landmarks[3], width, height)
    return (vertical1 + vertical2) / (2.0 * horizontal) if horizontal else 0
