import h5py
import torch
import torch.nn as nn
import torchvision 
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib as plt

# Load the Galaxy10 DECals dataset
with h5py.File('Galaxy10_DECals.h5', 'r') as f:
    images = f['images'][:]
    labels = f['ans'][:]

# Convert to float32 for memory efficiency and neural network compatibility
images = images.astype(np.float32)

# Calculate dataset statistics for normalization
mean_pixel = np.mean(images)
std_pixel = np.std(images)

# Custom normalization using dataset's mean and standard deviation
normalized_images = (images - mean_pixel) / std_pixel

# Create the training dataset as a tuple (images, labels)
training_dataset = (normalized_images, labels)

print(f"Dataset loaded and normalized successfully")
print(f"Images shape: {normalized_images.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Normalization - Mean: {mean_pixel:.6f}, Std: {std_pixel:.6f}")