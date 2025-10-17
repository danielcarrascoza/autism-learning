import numpy as np
import torch
from nn.model_nn import EngagementNN  # your trained NN architecture

class FusionModel:
    def __init__(self, student_id, model_path="engagement_nn.pth", device="cpu"):
        self.student_id = student_id
        self.device = device

        # Student personalization data
        self.student_data = {
            1: {"avg_score": 0.8, "preferred_mode": "Challenge"},
            2: {"avg_score": 0.6, "preferred_mode": "Standard"},
            3: {"avg_score": 0.4, "preferred_mode": "Simplified"}
        }

        # Load the trained neural network
        self.model = EngagementNN(3)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def combine(self, attention_now):
        """
        Combine gaze/blink/head_pose via trained neural network.
        Returns engagement score and adaptive mode.
        """
        features = np.array([
            attention_now["gaze_score"],
            attention_now["blink_score"],
            attention_now["head_pose_score"]
        ], dtype=np.float32)

        # Convert to tensor
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Predict engagement
        with torch.no_grad():
            engagement_pred = self.model(x).item()

        # Personalization
        baseline = self.student_data.get(self.student_id, {"avg_score": 0.5})["avg_score"]
        engagement = 0.7 * engagement_pred + 0.3 * baseline
        engagement = float(np.clip(engagement, 0, 1))

        # Choose adaptive mode
        if engagement > 0.75:
            mode = "Challenge Mode"
        elif engagement > 0.5:
            mode = "Standard Mode"
        else:
            mode = "Simplified Mode"

        return engagement, mode
