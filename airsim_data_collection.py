import airsim
import os
import numpy as np
import time
import cv2

# Connect to AirSim
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Take off
client.takeoffAsync().join()
time.sleep(2)

# Create folder to store depth images
SAVE_DIR = "airsim_data/depth"
os.makedirs(SAVE_DIR, exist_ok=True)

# Move forward and collect depth images
for i in range(300):
    client.moveByVelocityAsync(2, 0, 0, 1).join()

    # Request depth image
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, pixels_as_float=True)
    ])
    response = responses[0]

    if response is None or len(response.image_data_float) == 0:
        print(f"⚠️ Skipping frame {i} (empty depth image)")
        continue

    # Convert float array to 2D image
    img1d = np.array(response.image_data_float, dtype=np.float32)
    img2d = img1d.reshape(response.height, response.width)

    # Normalize and scale to 16-bit
    depth_clipped = np.clip(img2d, 0, 100)
    depth_normalized = (depth_clipped / 100.0 * 65535).astype(np.uint16)

    # Resize to 64x64 if needed
    depth_resized = cv2.resize(depth_normalized, (64, 64))

    # Save as 16-bit PNG
    filename = os.path.join(SAVE_DIR, f"depth_{i:04d}.png")
    cv2.imwrite(filename, depth_resized)
    print(f"✅ Saved depth image: {filename}")

    time.sleep(0.1)

# Land and reset
client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)

print("✅ Depth image data collection complete!")
