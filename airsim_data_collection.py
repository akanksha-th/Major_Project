import airsim
import os
import cv2
import numpy as np
import time

# Connect to AirSim
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Take off
client.takeoffAsync().join()

# Create folders for saving images
for cam in ["front", "left", "right", "bottom"]:
    os.makedirs(f"airsim_data/{cam}", exist_ok=True)

# Move forward and collect images
for i in range(300):  # Collect 300 images
    client.moveByVelocityAsync(2, 0, 0, 1).join()  # Slow forward movement

    # Capture images from 4 cameras
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),  # Front Camera
        airsim.ImageRequest("1", airsim.ImageType.Scene, False, False),  # Left Camera
        airsim.ImageRequest("2", airsim.ImageType.Scene, False, False),  # Right Camera
        airsim.ImageRequest("3", airsim.ImageType.Scene, False, False)   # Bottom Camera
    ])

    # Define camera labels
    cameras = ["front", "left", "right", "bottom"]

    for idx, response in enumerate(responses):
        # Convert image to numpy array
        img = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        img = img.reshape(response.height, response.width, 3)

        # Save image
        filename = f"airsim_data/{cameras[idx]}/frame_{i}.png"
        cv2.imwrite(filename, img)

        print(f"Saved {cameras[idx]} image: {filename}")

    time.sleep(0.1)  # Small delay

# Land the drone
client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)

print("✅ Data collection complete!")
