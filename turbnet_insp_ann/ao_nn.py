import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms

class AdaptiveOptics(nn.Module):
    def __init__(self):
        super(AdaptiveOptics, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(256, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = x.view(-1, 128 * 6 * 6)
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

model = AdaptiveOptics()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# Load data
data = np.load('data.npy')
labels = np.load('labels.npy')

# Convert data and labels to tensors
data = torch.from_numpy(data).float()
labels = torch.from_numpy(labels).float()

# Normalize the data
transform = transforms.Compose([transforms.Normalize((0.5,), (0.5,))])
data = transform(data)
# Define number of epochs
epochs = 100

for epoch in range(epochs):
    # Zero the gradients
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(data)
    
    # Compute loss
    loss = criterion(outputs, labels)
    
    # Backward pass
    loss.backward()
    
    # Update weights
    optimizer.step()
    
    # Print statistics
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, loss.item()))
    # Load test data
test_data = np.load('test_data.npy')
test_labels = np.load('test_labels.npy')

# Convert test data and labels to tensors
test_data = torch.from_numpy(test_data).float()
test_labels = torch.from_numpy(test_labels).float()

# Normalize test data
test_data = transform(test_data)