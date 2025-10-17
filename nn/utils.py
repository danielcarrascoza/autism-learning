import cv2
from mediapipe import solutions
import numpy as np

face_mesh = solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def compute_features_from_frame(frame):
    """
    Extracts [gaze_score, blink_score, head_pose_score] from a single frame
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return [0.0, 0.0, 0.0]

    for face_landmarks in results.multi_face_landmarks:
        
        left_eye = [face_landmarks.landmark[33], face_landmarks.landmark[133]]
        right_eye = [face_landmarks.landmark[362], face_landmarks.landmark[263]]
        nose_tip = face_landmarks.landmark[1]

        # Head pose proxy
        center_offset = abs(0.5 - nose_tip.x)
        head_pose_score = max(0, 1 - center_offset * 2)

        # Gaze proxy
        gaze_score = max(0, 1 - center_offset * 4)

        # Blink proxy
        left_eye_y = abs(left_eye[0].y - left_eye[1].y)
        right_eye_y = abs(right_eye[0].y - right_eye[1].y)
        blink_score = min(1.0, (left_eye_y + right_eye_y) * 25)

        return [gaze_score, blink_score, head_pose_score]

def normalize_features(X):
    """
    Standardize features: (x - mean) / std
    """
    X = np.array(X)
    return (X - X.mean(axis=0)) / X.std(axis=0)
