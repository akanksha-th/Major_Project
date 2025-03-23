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
time.sleep(2)  # Ensure drone stability

# Create folder for saving RGB images
os.makedirs("airsim_data/rgb", exist_ok=True)

# Move forward and collect images
for i in range(300):  # Collect 300 images
    client.moveByVelocityAsync(2, 0, 0, 1).join()  # Simple forward motion

    # Capture only the front camera RGB image
    response = client.simGetImage("0", airsim.ImageType.Scene)

    if response is None or len(response) == 0:
        print(f"⚠️ Skipping frame {i} (empty image)")
        continue  # Skip empty frames

    # Convert image to numpy array
    img_data = np.frombuffer(response, dtype=np.uint8)
    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)  # Decode image correctly

    if img is None:
        print(f"⚠️ Failed to decode frame {i}")
        continue

    # Save image
    filename = f"airsim_data/rgb/frame_{i}.png"
    cv2.imwrite(filename, img)
    print(f"✅ Saved RGB image: {filename}")

    time.sleep(0.1)  # Small delay

# Land the drone
client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)

print("✅ Data collection complete!")
