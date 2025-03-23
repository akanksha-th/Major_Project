import airsim
import numpy as np
import torch
import time
from model import DDDQN
import cv2

# Initialize AirSim client
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)
client.setCarControls(airsim.CarControls())

# Load trained model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DDDQN().to(DEVICE)
model.load_state_dict(torch.load("dd_dqn_object_avoidance.pth"))
model.eval()

# Action mapping: 0 = left, 1 = right, 2 = forward
ACTIONS = {
    0: (-0.5, 0.0),  # Left (steering, throttle)
    1: (0.5, 0.0),   # Right
    2: (0.0, 1.0)    # Forward
}

def preprocess_image(response):
    """ Convert AirSim image response to model input. """
    img = np.frombuffer(response.image_data_uint8, dtype=np.uint8).reshape(response.height, response.width, 3)
    img = cv2.resize(img, (64, 64)) / 255.0  # Resize and normalize
    img = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(DEVICE)  # (C, H, W)
    return img

# Run simulation loop
for step in range(500):
    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
    state = preprocess_image(responses[0])
    
    with torch.no_grad():
        q_values = model(state)
        action = torch.argmax(q_values).item()

    steering, throttle = ACTIONS[action]
    controls = airsim.CarControls()
    controls.steering = steering
    controls.throttle = throttle
    client.setCarControls(controls)

    print(f"Step {step}: Action {action} (Steering={steering}, Throttle={throttle})")
    time.sleep(0.1)

# Reset AirSim
client.reset()
print("✅ Simulation Complete!")
