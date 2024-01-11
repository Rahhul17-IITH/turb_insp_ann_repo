#deep image prior method
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np

# Define the DIP model
class DIPModel(nn.Module):
    def __init__(self):
        super(DIPModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x

# Define a custom dataset
class TurbulenceDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.scenes = sorted(os.listdir(root_dir))

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        scene_dir = os.path.join(self.root_dir, self.scenes[idx], 'turb')
        turb_images = sorted(os.listdir(scene_dir))
        turb_path = os.path.join(scene_dir, turb_images[0])
        clear_path = os.path.join(self.root_dir, self.scenes[idx], 'gt.png')

        turb_image = Image.open(turb_path).convert("RGB")
        clear_image = Image.open(clear_path).convert("RGB")

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        turb_image = transform(turb_image)
        clear_image = transform(clear_image)

        return turb_image, clear_image

# Function to calculate PSNR
def calculate_psnr(original, restored):
    mse = np.mean((original - restored) ** 2)
    max_pixel = 1.0  # Assuming pixel values are in the range [0, 1]
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# Define training parameters
batch_size = 16
learning_rate = 0.001
num_epochs = 10

# Create the DIP model, loss function, and optimizer
model = DIPModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Create the dataset and data loader
train_dataset = TurbulenceDataset(root_dir='heatchamber_new')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    for turb_images, clear_images in train_loader:
        # Forward pass
        outputs = model(turb_images)

        # Compute the loss
        loss = criterion(outputs, clear_images)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Test the model on the test image
test_image_path = 'test_input_image/28279217.png'
test_image = Image.open(test_image_path).convert("RGB")
transform = transforms.Compose([transforms.ToTensor()])
test_image = transform(test_image).unsqueeze(0)  # Add batch dimension

# Perform inference on the test image
with torch.no_grad():
    restored_image = model(test_image)

# Convert tensors to numpy arrays
test_image_np = test_image.squeeze(0).permute(1, 2, 0).numpy()
restored_image_np = restored_image.squeeze(0).permute(1, 2, 0).numpy()

# Calculate PSNR
psnr_value = calculate_psnr(test_image_np, restored_image_np)
print(f'PSNR: {psnr_value:.2f}')

# Save the restored image
restored_image = (restored_image_np * 255).astype('uint8')
restored_image = Image.fromarray(restored_image)
restored_image.save('restored_image_dip.png')






