import torch
from nn.model_nn import EngagementNN

# Initialize model with same architecture
model = EngagementNN(input_dim=3)

# Load trained weights
model.load_state_dict(torch.load("engagement_nn.pth"))
model.eval()

# fc1 = first linear layer
fc1_weights = model.fc1.weight.data  # shape: [128, 3]
fc1_bias = model.fc1.bias.data       # shape: [128]

print("First layer weights (3 features → 128 neurons):")
print(fc1_weights)

# Optional: see contribution of each input feature by averaging across neurons
avg_feature_weight = fc1_weights.abs().mean(dim=0)
print("Average absolute weight per input feature:")
print(f"Gaze: {avg_feature_weight[0]:.4f}, Blink: {avg_feature_weight[1]:.4f}, HeadPose: {avg_feature_weight[2]:.4f}")

influence = fc1_weights.abs().sum(dim=0)
print("Total influence of each input feature:")
print(f"Gaze: {influence[0]:.4f}, Blink: {influence[1]:.4f}, HeadPose: {influence[2]:.4f}")