import h5py
import numpy as np

# Load the compressed Galaxy10 dataset
with h5py.File('Galaxy10_small.h5', 'r') as f:
    # Load all images from the compressed dataset
    images = f['images'][:]
    labels = f['ans'][:]
    total_images = len(images)

# Convert to float32 for memory efficiency and neural network compatibility
images = images.astype(np.float32)

# Calculate dataset statistics for normalization (using the subset)
mean_pixel = np.mean(images)
std_pixel = np.std(images)

# Custom normalization using dataset's mean and standard deviation
normalized_images = (images - mean_pixel) / std_pixel

# Create the training dataset as a tuple (images, labels)
training_dataset = (normalized_images, labels)

# Calculate actual size in MB
actual_size_mb = (normalized_images.nbytes + labels.nbytes) / (1024 * 1024)

print(f"Dataset loaded and normalized successfully")
print(f"Using {total_images} samples")
print(f"Images shape: {normalized_images.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Actual dataset size: {actual_size_mb:.1f} MB")
print(f"Normalization - Mean: {mean_pixel:.6f}, Std: {std_pixel:.6f}")