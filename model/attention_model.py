import cv2
import numpy as np
from mediapipe import solutions

class AttentionDetector:
    def __init__(self):
        self.face_mesh = solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def analyze_frame(self, frame):
        """
        Returns attention metrics (dict) and heatmap overlay.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        h, w, _ = frame.shape
        heatmap = np.zeros((h, w), dtype=np.float32)

        attention_now = {
            "gaze_score": 0.0,
            "blink_score": 0.0,
            "head_pose_score": 0.0
        }

        if not results.multi_face_landmarks:
            # No face detected → return default low attention
            return attention_now, frame

        for face_landmarks in results.multi_face_landmarks:
            # Eye and head key points
            left_eye = [face_landmarks.landmark[33], face_landmarks.landmark[133]]
            right_eye = [face_landmarks.landmark[362], face_landmarks.landmark[263]]
            nose_tip = face_landmarks.landmark[1]

            # Convert to pixel coordinates
            lx = int(left_eye[0].x * w)
            rx = int(right_eye[0].x * w)
            ny = int(nose_tip.y * h)

            # Gaze/attention metrics
            center_offset = abs(0.5 - nose_tip.x)
            gaze_score = max(0, 1 - center_offset * 4)

            left_eye_y = abs(left_eye[0].y - left_eye[1].y)
            right_eye_y = abs(right_eye[0].y - right_eye[1].y)
            blink_score = min(1.0, (left_eye_y + right_eye_y) * 25)

            attention_now = {
                "gaze_score": float(gaze_score),
                "blink_score": float(blink_score),
                "head_pose_score": 1.0 - center_offset * 2
            }

            # Draw heatmap
            for point in [left_eye[0], right_eye[0], nose_tip]:
                px, py = int(point.x * w), int(point.y * h)
                cv2.circle(heatmap, (px, py), 25, 1, -1)

        # Apply color map
        heatmap = cv2.GaussianBlur(heatmap, (45, 45), 0)
        heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(frame, 0.6, heatmap_color, 0.4, 0)

        return attention_now, overlay
