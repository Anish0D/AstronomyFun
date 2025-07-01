import h5py
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as f
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

## Using my CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with h5py.File('Galaxy10_DECals.h5', 'r') as f:
    images = f['images'][:]   # Shape: (17736, 256, 256, 3)
    labels = f['ans'][:]      # Shape: (17736,)

# Convert images to float32 for processing
images = images.astype(np.float32)

# Compute dataset mean and std for normalization
mean_pixel = np.mean(images)     
std_pixel = np.std(images)

class GalaxyDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]  # (256, 256, 3)
        label = self.labels[idx]

        image = Image.fromarray(image.astype(np.uint8))

        if self.transform:
            image = self.transform(image)

        return image, label

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),              
    transforms.RandomRotation(degrees=15),          
    transforms.ToTensor(),                         
    transforms.Normalize(                           
        mean=[mean_pixel / 255] * 3,                
        std=[std_pixel / 255] * 3
    )
])

galaxy_dataset = GalaxyDataset(images, labels, transform=transform)
train_loader = DataLoader(galaxy_dataset, batch_size=64, shuffle=True)

class GCNN(nn.Module):
    def __init__(self, classes=10):
        super(GCNN, self).__init__()
        ## I want to implement 3 Convolution layers to make the model identify specific features like arms, etc
        ## Plus 2 pooling layers 
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))

        ## Applying 2 hidden and 1 output layers for the data to flow through
        ## the dropout command is in order to strengthen the NN, by forcing all neurons to be relied on
        ## it also allows for the NN to learn more specific features!!
        self.fc1 = nn.Linear(256*8*8, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)
    def forward(self, x):
        ## Defining the usual forward function to move the data through the neural network
        x = f.relu(self.conv1(x))
        x = f.relu(self.conv2(x))
        x = self.pool(x)
        x = f.relu(self.conv3(x))
        x = self.adaptive_pool(x)

        x = x.view(x.size(0), -1)
        x = f.relu(self.fc1(x))
        x = self.dropout(x)
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x

## Loss function and optimizer
model = GCNN().to(device)
loss_calc = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

## Setting up the Training YAY
model.train()

## Training Loop :)
accuracies = []
for epoch in range(0, 31):
     # keeping track of loss
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = loss_calc(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(outputs, 1)         # get the index of the highest score for each image
        correct += (predicted == labels).sum().item()  # count how many were right
        total += labels.size(0)                       # total images in this batch


    accuracy = 100 * correct / total
    accuracies.append(accuracy)
    print(f"Epoch {epoch+1}, Loss:{total_loss:.4f}, Accuracy:{accuracy:.2f}%")

# Save the trained model
torch.save(model.state_dict(), 'GalaxyID.pth')
print("Model saved as GalaxyID.pth")

# Graphing accuracy growth
plt.plot(range(1, 31), accuracies)
plt.ylabel('Accuracy in %')
plt.xlabel('Number of epochs run')
plt.show()