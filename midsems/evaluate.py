import airsim
import numpy as np
import torch
import time
from model import DDDQN
import matplotlib.pyplot as plt
import cv2

# Initialize AirSim
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

print("🚁 Taking off...")
client.takeoffAsync().join()
client.moveToZAsync(-3, 1).join()
time.sleep(2)

# Load trained model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DDDQN(input_channels=1, num_actions=3).to(DEVICE)
model.load_state_dict(torch.load("dd_dqn_object_avoidance.pth", map_location=DEVICE))
model.eval()

ACTIONS = {
    0: (-1.0, 0.0, 0.0),  # Move left
    1: (1.0, 0.0, 0.0),   # Move right
    2: (0.0, 1.5, 0.0)    # Move forward
}
DURATION = 1.0

def preprocess_depth_image(response):
    img1d = np.array(response.image_data_float, dtype=np.float32)
    img2d = img1d.reshape(response.height, response.width)
    img2d = cv2.resize(img2d, (64, 64))
    img2d = np.clip(img2d, 0, 100) / 100.0
    depth_tensor = torch.tensor(img2d, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    return depth_tensor, img2d

# Path tracking
path_x, path_y = [], []

# Live plot
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

# Main loop
for step in range(5000):
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True)
    ])
    state, raw_depth = preprocess_depth_image(responses[0])

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
    client.moveByVelocityAsync(vx, vy, vz, DURATION).join()

    position = client.getMultirotorState().kinematics_estimated.position
    path_x.append(position.x_val)
    path_y.append(position.y_val)
    update_plot()

    print(f"[{step}] Action {action}: vx={vx}, vy={vy}, vz={vz}")
    time.sleep(0.1)

np.save("flight_path.npy", np.array([path_x, path_y]))
print("📍 Path saved to 'flight_path.npy'")

client.hoverAsync().join()
client.armDisarm(False)
client.enableApiControl(False)
plt.ioff()
plt.show()
print("✅ Drone simulation complete!")
