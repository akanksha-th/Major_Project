import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import matplotlib.pyplot as plt
from collections import deque

# Hyperparameters
IMG_SHAPE = (3, 64, 64)
LEARNING_RATE = 0.0001
GAMMA = 0.99
MEMORY_SIZE = 5000
BATCH_SIZE = 32
EPISODES = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load processed images
DATASET_PATH = "processed_data"
image_files = sorted(os.listdir(DATASET_PATH))
replay_memory = deque(maxlen=MEMORY_SIZE)

# Load images into memory for training
for file in image_files:
    img = np.load(os.path.join(DATASET_PATH, file)) / 255.0  # Normalize
    state = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32)
    action = random.choice([0, 1, 2])  # 3 actions
    reward = 1
    replay_memory.append((state, action, reward))

# Define Model
class DDDQN(nn.Module):
    def __init__(self):
        super(DDDQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)  # No softmax for CrossEntropyLoss

# Initialize model, optimizer, loss function
model = DDDQN().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# Tracking performance
losses = []
q_values = []

# Train DD-DQN model
for episode in range(EPISODES):
    batch = random.sample(replay_memory, BATCH_SIZE)
    states, actions, rewards = zip(*batch)

    states = torch.stack(states).to(DEVICE)
    actions = torch.tensor(actions, dtype=torch.long).to(DEVICE)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)

    # Forward pass
    outputs = model(states)

    # Compute Q-values
    target_q_values = rewards + GAMMA * torch.max(outputs, dim=1)[0].detach()

    # Compute loss
    loss = criterion(outputs, actions)
    losses.append(loss.item())

    # Store max Q-value for visualization
    q_values.append(torch.max(outputs).item())

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if episode % 10 == 0:
        print(f"Episode {episode}/{EPISODES} - Loss: {loss.item():.4f} - Max Q-value: {q_values[-1]:.4f}")

# Save trained model
torch.save(model.state_dict(), "dd_dqn_object_avoidance.pth")
print("✅ Model training complete! Saved as 'dd_dqn_object_avoidance.pth'")

# Plot training loss
plt.figure(figsize=(10, 5))
plt.plot(losses, label="Loss")
plt.xlabel("Episodes")
plt.ylabel("Loss")
plt.title("Training Loss over Episodes")
plt.legend()
plt.savefig("demo_flight/training_loss.png")
plt.close()

# Plot Q-values
plt.figure(figsize=(10, 5))
plt.plot(q_values, label="Max Q-Value")
plt.xlabel("Episodes")
plt.ylabel("Q-Value")
plt.title("Max Q-Value per Episode")
plt.legend()
plt.savefig("demo_flight/q_values.png")
plt.close()

print("Training Plots saved!!")