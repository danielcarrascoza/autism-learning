import torch
from nn import EngagementNN, train_nn, normalize_features
from data.loader import load_dataset

# Load dataset
X, y = load_dataset("/Volumes/extra storage/DAiSEE/Labels/TrainLabels.csv", "/Volumes/extra storage/DAiSEE/DataSet/Train/")
X_norm = normalize_features(X)
X_tensor = torch.tensor(X_norm, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# Initialize and train model
model = EngagementNN(input_dim=3)
model = train_nn(model, X_tensor, y_tensor, epochs=10)

# Save model
torch.save(model.state_dict(), "engagement_nn.pth")
print("Model trained and saved!")
