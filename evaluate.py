import airsim
import numpy as np
import torch
import time
from model import DDDQN
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
    0: (-0.5, 0.0),  # Left
    1: (0.5, 0.0),   # Right
    2: (0.0, 1.0)    # Forward
}


airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True)

def preprocess_depth_image(response):
    img1d = np.array(response.image_data_float, dtype=np.float32)
    img2d = img1d.reshape(response.height, response.width)
    img2d = cv2.resize(img2d, (64, 64))
    img2d = np.expand_dims(img2d, axis=0)  # (1, 64, 64)
    return torch.tensor(img2d, dtype=torch.float32).unsqueeze(0).to(DEVICE)


# Run simulation loop
for step in range(500):
    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
    state = preprocess_depth_image(responses[0])
    
    with torch.no_grad():
        q_values = model(state)
        action = torch.argmax(q_values).item()

    steering, throttle = ACTIONS[action]
    controls = airsim.CarControls()
    controls.steering = steering
    controls.throttle = throttle
    client.moveByVelocityAsync(vx, vy, vz, duration)


    print(f"Step {step}: Action {action} (Steering={steering}, Throttle={throttle})")
    time.sleep(0.1)

# Reset AirSim
client.reset()
print("✅ Simulation Complete!")
