import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import cv2
import numpy as np
from collections import deque
from tqdm import tqdm

# Constants
DATASET_PATH = "airsim_data/depth"
SAVE_PATH = "dd_dqn_object_avoidance.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_ACTIONS = 3  # Forward, Left, Right
INPUT_CHANNELS = 1  # Depth input

# Hyperparameters
BATCH_SIZE = 32
GAMMA = 0.99
LR = 1e-4
MEMORY_SIZE = 5000
TARGET_UPDATE_FREQ = 10
NUM_EPOCHS = 5000

# ---- Model Definition: Dueling Double DQN ---- #
import torch
import torch.nn as nn
import torch.nn.functional as F

class DDDQN(nn.Module):
    def __init__(self, input_channels=1, num_actions=3):
        super(DDDQN, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Calculate the size after conv layers
        self.flattened_size = self._get_conv_output((1, 64, 64))  # assuming input is 64x64

        self.fc = nn.Linear(self.flattened_size, 512)

        # Advantage and value streams
        self.fc_adv = nn.Linear(512, num_actions)
        self.fc_val = nn.Linear(512, 1)

    def _get_conv_output(self, shape):
        with torch.no_grad():
            x = torch.zeros(1, *shape)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            return int(torch.flatten(x, 1).shape[1])

    def forward(self, x):
        x = F.relu(self.conv1(x))  # [B, 32, H/4, W/4]
        x = F.relu(self.conv2(x))  # [B, 64, H/8, W/8]
        x = F.relu(self.conv3(x))  # [B, 64, H/8 - 2, W/8 - 2]

        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc(x))

        adv = self.fc_adv(x)
        val = self.fc_val(x).expand(x.size(0), adv.size(1))

        return val + adv - adv.mean(1, keepdim=True)


# ---- Experience Replay ---- #
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state = zip(*batch)
        return torch.stack(state), torch.tensor(action), torch.tensor(reward), torch.stack(next_state)

    def __len__(self):
        return len(self.buffer)

# ---- Load dataset from .pt files ---- #

def load_dataset(path):
    data = []
    files = sorted(os.listdir(path))
    for i in range(len(files) - 1):
        current_img = cv2.imread(os.path.join(path, files[i]), cv2.IMREAD_UNCHANGED)
        next_img = cv2.imread(os.path.join(path, files[i + 1]), cv2.IMREAD_UNCHANGED)

        if current_img is None or next_img is None:
            continue

        # Normalize and convert to tensor
        current = torch.tensor(current_img, dtype=torch.float32).unsqueeze(0) / 255.0  # [1, H, W]
        next_state = torch.tensor(next_img, dtype=torch.float32).unsqueeze(0) / 255.0

        action = 0  # Dummy forward action
        reward = 1.0

        data.append((current, action, reward, next_state))

    return data


# ---- Training Function ---- #
def train():
    # Init networks
    policy_net = DDDQN(INPUT_CHANNELS, NUM_ACTIONS).to(DEVICE)
    target_net = DDDQN(INPUT_CHANNELS, NUM_ACTIONS).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    replay_buffer = ReplayBuffer(MEMORY_SIZE)

    # Load and shuffle dataset
    dataset = load_dataset(DATASET_PATH)
    print(f"📦 Loaded {len(dataset)} frames from {DATASET_PATH}")
    random.shuffle(dataset)

    # Populate replay buffer
    for transition in dataset:
        replay_buffer.push(*transition)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        if len(replay_buffer) < BATCH_SIZE:
            continue

        state_batch, action_batch, reward_batch, next_state_batch = replay_buffer.sample(BATCH_SIZE)

        state_batch = state_batch.to(DEVICE)
        next_state_batch = next_state_batch.to(DEVICE)
        action_batch = action_batch.to(DEVICE)
        reward_batch = reward_batch.to(DEVICE).float()

        q_values = policy_net(state_batch)
        next_q_values = target_net(next_state_batch)

        q_value = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)
        max_next_q_value = next_q_values.max(1)[0]

        expected_q = reward_batch + GAMMA * max_next_q_value

        loss = F.mse_loss(q_value, expected_q)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())
            print(f"🔁 Target network updated at epoch {epoch + 1}")

        print(f"📚 Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {loss.item():.4f}")

    # Save trained model
    torch.save(policy_net.state_dict(), SAVE_PATH)
    print(f"✅ Model saved to {SAVE_PATH}")

# Run training
if __name__ == "__main__":
    train()
