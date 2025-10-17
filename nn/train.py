import torch
import torch.nn as nn
import torch.optim as optim

train_losses = []

def train_nn(model, X, y, epochs=10, batch_size=32, lr=0.001):
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        running_loss = 0.0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(loader):.4f}")

    import matplotlib.pyplot as plt

    plt.plot(train_losses)
    plt.title("Training Loss Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    return model
