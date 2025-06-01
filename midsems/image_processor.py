import os
import cv2
import numpy as np
import torch

# Paths
DATASET_PATH = "airsim_data/depth"          # Folder with raw AirSim depth .png images
PROCESSED_PATH = "processed_data/tensors"   # Output directory for processed .pt files

# Create folder for processed tensors
os.makedirs(PROCESSED_PATH, exist_ok=True)

# Target image shape
IMG_SIZE = (64, 64)
MAX_DEPTH_METERS = 100.0  # Clipping threshold, must match simulation normalization

for filename in os.listdir(DATASET_PATH):
    if not filename.endswith(".png"):
        continue

    img_path = os.path.join(DATASET_PATH, filename)

    # Load depth image as float32 (grayscale)
    depth_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    if depth_img is None:
        print(f"⚠️ Failed to load: {filename}")
        continue

    depth_img = depth_img.astype(np.float32)

    # Resize to (64, 64) and normalize
    depth_img = cv2.resize(depth_img, IMG_SIZE)
    depth_img = np.clip(depth_img, 0, MAX_DEPTH_METERS) / MAX_DEPTH_METERS  # Normalize to [0, 1]

    # Add channel dimension: (1, 64, 64)
    depth_tensor = torch.tensor(depth_img).unsqueeze(0).unsqueeze(0).float()  # shape: [1, 1, 64, 64]

    # Save as .pt (PyTorch tensor)
    save_name = filename.replace(".png", ".pt")
    torch.save(depth_tensor, os.path.join(PROCESSED_PATH, save_name))

    print(f"✅ Processed and saved: {save_name}")

print("🎉 All depth images processed and saved as torch tensors in 'processed_data/tensors/'")
