import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import gym
from collections import deque

# Hyperparameters
IMG_SHAPE = (3, 64, 64)  # PyTorch uses (Channels, Height, Width)
LEARNING_RATE = 0.0001
GAMMA = 0.99
MEMORY_SIZE = 5000
BATCH_SIZE = 32
EPISODES = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load processed images
DATASET_PATH = "processed_data"
image_files = sorted(os.listdir(DATASET_PATH))

# Create replay buffer
replay_memory = deque(maxlen=MEMORY_SIZE)

# Load images into memory for training
for file in image_files:
    img = np.load(os.path.join(DATASET_PATH, file)) / 255.0  # Normalize
    state = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32)  # Convert to (C, H, W)
    action = random.choice([0, 1, 2])  # Example: 0 = left, 1 = right, 2 = forward
    reward = 1  # Reward for moving forward without collision
    replay_memory.append((state, action, reward))

# Define DD-DQN Model in PyTorch
class DDDQN(nn.Module):
    def __init__(self):
        super(DDDQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)  # Assuming (64, 16, 16) after conv layers
        self.fc2 = nn.Linear(128, 3)  # 3 possible actions

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=1)  # Output action probabilities

# Initialize model
model = DDDQN().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# Train DD-DQN model
for episode in range(EPISODES):
    batch = random.sample(replay_memory, BATCH_SIZE)
    states, actions, rewards = zip(*batch)

    states = torch.stack(states).to(DEVICE)
    actions = torch.tensor(actions, dtype=torch.long).to(DEVICE)

    # Forward pass
    outputs = model(states)
    loss = criterion(outputs, actions)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if episode % 10 == 0:
        print(f"Episode {episode}/{EPISODES} completed! Loss: {loss.item():.4f}")

# Save trained model
torch.save(model.state_dict(), "dd_dqn_object_avoidance.pth")
print("✅ Model training complete! Saved as 'dd_dqn_object_avoidance.pth'")
