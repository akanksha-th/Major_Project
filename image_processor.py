import os
import cv2
import numpy as np

# Paths
DATASET_PATH = "airsim_data/rgb"
PROCESSED_PATH = "processed_data"

# Create folder for processed images
os.makedirs(PROCESSED_PATH, exist_ok=True)

# Image size for training (match RL model input)
IMG_SIZE = (64, 64)

for filename in os.listdir(DATASET_PATH):
    img_path = os.path.join(DATASET_PATH, filename)

    # Read, resize, and normalize image
    img = cv2.imread(img_path)
    img = cv2.resize(img, IMG_SIZE)  # Resize to 64x64
    img = img / 255.0  # Normalize pixel values (0-1)

    # Save processed image
    np.save(os.path.join(PROCESSED_PATH, filename.replace(".png", ".npy")), img)

print("✅ Image preprocessing complete! Images saved in 'processed_data/'")
