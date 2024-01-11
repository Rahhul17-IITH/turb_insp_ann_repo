import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms

# Define the neural network architecture
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

# Define the transform for normalizing the data
model = AdaptiveOptics()
transform = transforms.Compose([transforms.Normalize((0.5,), (0.5,))])

# Load the training data and labels
train_data = np.load('train_data.npy')
train_labels = np.load('train_labels.npy')

# Convert the training data and labels to tensors
train_data = torch.from_numpy(train_data).float()
train_labels = torch.from_numpy(train_labels).float()

# Normalize the training data
train_data = transform(train_data)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for i in range(len(train_data)):
        # Forward pass
        outputs = model(train_data[i].unsqueeze(0))
        loss = criterion(outputs, train_labels[i].unsqueeze(0))

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, running_loss/len(train_data)))

# Save the trained model
torch.save(model.state_dict(), 'adaptive_optics_model.pth')
