import pandas as pd

class FusionModel:
    def __init__(self, student_id, csv_path="data/students.csv"):
        self.students = pd.read_csv(csv_path)
        self.student = self.students[self.students["student_id"] == student_id].iloc[0]
        # Research-inspired weights
        self.weights = {
            "gaze_score": 0.5,
            "blink_rate": -0.3,
            "emotion_engagement": 0.4,
            "avg_test_score": 0.2,
            "visual_preference": 0.4,
            "attention_consistency": 0.3
        }

    def combine(self, attention_now):
        s = self.student
        engagement = (
            attention_now["gaze_score"] * self.weights["gaze_score"] +
            attention_now["blink_rate"] * self.weights["blink_rate"] +
            attention_now["emotion_engagement"] * self.weights["emotion_engagement"] +
            (s["avg_test_score"]/100) * self.weights["avg_test_score"] +
            s["visual_preference"] * self.weights["visual_preference"] +
            s["attention_consistency"] * self.weights["attention_consistency"]
        )
        engagement = max(0, min(engagement, 1))
        if engagement > 0.6:
            mode = "Challenge Mode 🧩"
        elif engagement > 0.4:
            mode = "Standard Lesson 📘"
        else:
            mode = "Simplified Visual Lesson 🎨"
        return engagement, mode
