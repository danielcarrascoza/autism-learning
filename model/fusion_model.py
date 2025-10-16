import numpy as np

class FusionModel:
    def __init__(self, student_id):
        self.student_id = student_id
        self.student_data = {
            1: {"avg_score": 0.8, "preferred_mode": "Challenge"},
            2: {"avg_score": 0.6, "preferred_mode": "Standard"},
            3: {"avg_score": 0.4, "preferred_mode": "Simplified"}
        }
        self.weights = {
            "gaze_score": 0.5,
            "blink_score": 0.3,
            "head_pose_score": 0.2
        }

    def combine(self, attention_now):
        """
        Combine multiple attention metrics with prior student data.
        Returns engagement score and adaptive mode.
        """
        # Weighted fusion of attention metrics
        engagement = 0
        for key, value in attention_now.items():
            weight = self.weights.get(key, 0)
            engagement += value * weight

        baseline = self.student_data.get(self.student_id, {"avg_score": 0.5})["avg_score"]
        engagement = 0.7 * engagement + 0.3 * baseline
        engagement = float(np.clip(engagement, 0, 1))

        # Decide adaptive lesson mode
        if engagement > 0.75:
            mode = "Challenge Mode"
        elif engagement > 0.5:
            mode = "Standard Mode"
        else:
            mode = "Simplified Mode"

        return engagement, mode
