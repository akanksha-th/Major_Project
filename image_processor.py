import os
import numpy as np
import cv2

# Paths
DATASET_PATH = "airsim_data/depth"
PROCESSED_PATH = "processed_data"

# Create folder for processed images
os.makedirs(PROCESSED_PATH, exist_ok=True)

# Image size for training (match RL model input)
IMG_SIZE = (64, 64)

for filename in os.listdir(DATASET_PATH):
    if filename.endswith(".npy"):
        depth_array = np.load(os.path.join(DATASET_PATH, filename))  # Already float32 depth values

        # Resize to 64x64 and clip extreme values
        depth_array = cv2.resize(depth_array, IMG_SIZE)
        depth_array = np.clip(depth_array, 0, 100)  # Clip max depth to 100m

        # Normalize depth to 0–1
        depth_array /= 100.0

        # Add channel dimension (1, 64, 64)
        depth_array = np.expand_dims(depth_array, axis=0)

        # Save preprocessed
        save_path = os.path.join(PROCESSED_PATH, filename)
        np.save(save_path, depth_array)

print("✅ Depth image preprocessing complete!")
