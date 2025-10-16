import cv2
import mediapipe as mp
import numpy as np

mp_face = mp.solutions.face_mesh

class AttentionDetector:
    def __init__(self):
        self.face_mesh = mp_face.FaceMesh(refine_landmarks=True)
        self.last_gaze_score = 0.5

    def analyze_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            return {"gaze_score": 0.0, "blink_rate": 0.5, "emotion_engagement": 0.5}

        face_landmarks = results.multi_face_landmarks[0]
        landmarks = np.array([(lm.x, lm.y) for lm in face_landmarks.landmark])

        # Approximate gaze: stability in x direction
        gaze_score = np.clip(1 - np.std(landmarks[:, 0]) * 10, 0, 1)
        # Fake blink rate & emotion proxies for MVP
        blink_rate = np.random.uniform(0.2, 0.5)
        emotion_engagement = np.random.uniform(0.6, 0.9)

        return {
            "gaze_score": float(gaze_score),
            "blink_rate": float(blink_rate),
            "emotion_engagement": float(emotion_engagement)
        }
