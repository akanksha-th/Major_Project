import airsim
import numpy as np
import torch
import time
from model import DDDQN
import matplotlib.pyplot as plt
import cv2

# Initialize AirSim client
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()
client.moveByVelocityAsync(0, 0, 0, 1).join()  # small hover


# Load trained model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DDDQN().to(DEVICE)
model.load_state_dict(torch.load("dd_dqn_object_avoidance.pth"))
model.eval()

ACTIONS = {
    0: (-1.0, 0.0, 0.0),  # Move left (negative X)
    1: (1.0, 0.0, 0.0),   # Move right (positive X)
    2: (0.0, 1.5, 0.0)    # Move forward (positive Y)
}
DURATION = 1.0

def preprocess_depth_image(response):
    img1d = np.array(response.image_data_float, dtype=np.float32)
    img2d = img1d.reshape(response.height, response.width)
    img2d = cv2.resize(img2d, (64, 64))
    img2d = np.clip(img2d, 0, 100)
    img2d /= 100.0
    depth_tensor = np.expand_dims(img2d, axis=0)
    return torch.tensor(depth_tensor, dtype=torch.float32).unsqueeze(0).to(DEVICE), img2d

# === Path recording ===
path_x, path_y = [], []

# === Live plot setup ===
plt.ion()
fig, ax = plt.subplots()
scat, = ax.plot([], [], 'bo-')
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 30)
ax.set_title("Drone Path (Top View)")
ax.set_xlabel("X")
ax.set_ylabel("Y")

def update_plot():
    scat.set_data(path_x, path_y)
    fig.canvas.draw()
    fig.canvas.flush_events()

# Run simulation loop
for step in range(500):
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True)
        ])
    state, raw_depth = preprocess_depth_image(responses[0])
    
    # Crash detection: too close or actual collision
    if np.min(raw_depth) < 1.0:
        print(f"[{step}] ❗ Too close to obstacle. Aborting.")
        break

    collision_info = client.getMultirotorState().collision
    if collision_info.has_collided:
        print(f"[{step}] 💥 Collision detected! Ending simulation.")
        break
    
    with torch.no_grad():
        q_values = model(state)
        action = torch.argmax(q_values).item()

    vx, vy, vz = ACTIONS[action]
    client.moveByVelocityAsync(vx, vy, vz, DURATION)
    
    # Track position
    position = client.getMultirotorState().kinematics_estimated.position
    path_x.append(position.x_val)
    path_y.append(position.y_val)
    update_plot()

    print(f"[{step}] Action {action}: vx={vx}, vy={vy}, vz={vz}")
    time.sleep(0.1)

np.save("flight_path.npy", np.array([path_x, path_y]))
print("📍 Path saved to 'flight_path.npy'")

# Reset AirSim
client.hoverAsync().join()
client.armDisarm(False)
client.enableApiControl(False)
plt.ioff()
plt.show()
print("✅ Drone simulation complete!")
