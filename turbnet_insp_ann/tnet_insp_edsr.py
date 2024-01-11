# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset
# from torchvision import transforms
# from PIL import Image
# import os
# import numpy as np
# from torchsr.models import edsr

# # Define a custom dataset
# class TurbulenceDataset(Dataset):
#     def __init__(self, root_dir):
#         self.root_dir = root_dir
#         self.scenes = sorted(os.listdir(root_dir))

#     def __len__(self):
#         return len(self.scenes)

#     def __getitem__(self, idx):
#         scene_dir = os.path.join(self.root_dir, self.scenes[idx], 'turb')
#         turb_images = sorted(os.listdir(scene_dir))
#         turb_path = os.path.join(scene_dir, turb_images[0])
#         clear_path = os.path.join(self.root_dir, self.scenes[idx], 'gt.png')

#         turb_image = Image.open(turb_path).convert("RGB")
#         clear_image = Image.open(clear_path).convert("RGB")

#         transform = transforms.Compose([
#             transforms.ToTensor(),
#         ])

#         turb_image = transform(turb_image)
#         clear_image = transform(clear_image)

#         return turb_image, clear_image

# # Function to calculate PSNR
# def calculate_psnr(original, restored):
#     mse = np.mean((original - restored) ** 2)
#     max_pixel = 1.0  # Assuming pixel values are in the range [0, 1]
#     psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
#     return psnr

# # Define training parameters
# batch_size = 16
# learning_rate = 0.001
# num_epochs = 10

# # Create the model, loss function, and optimizer
# model = edsr(num_layers=32, scale=2)  # You can adjust num_layers and scale according to your requirements
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# # Create the dataset and data loader
# train_dataset = TurbulenceDataset(root_dir='heatchamber_new')
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# # Training loop
# for epoch in range(num_epochs):
#     for turb_images, clear_images in train_loader:
#         # Forward pass
#         outputs = model(turb_images)

#         # Compute the loss
#         loss = criterion(outputs, clear_images)

#         # Backward pass and optimization
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# # Save the trained model
# torch.save(model.state_dict(), 'image_restoration_model.pth')

# # Test the model on the test image
# test_image_path = 'test_input_image/28279217.png'
# test_image = Image.open(test_image_path).convert("RGB")
# transform = transforms.Compose([transforms.ToTensor()])
# test_image = transform(test_image).unsqueeze(0)  # Add batch dimension

# # Load the trained model
# model.load_state_dict(torch.load('image_restoration_model.pth'))
# model.eval()

# # Perform inference on the test image
# with torch.no_grad():
#     restored_image = model(test_image)

# # Convert tensors to numpy arrays
# test_image_np = test_image.squeeze(0).permute(1, 2, 0).numpy()
# restored_image_np = restored_image.squeeze(0).permute(1, 2, 0).numpy()

# # Calculate PSNR
# psnr_value = calculate_psnr(test_image_np, restored_image_np)
# print(f'PSNR: {psnr_value:.2f}')

# # Save the restored image
# restored_image = (restored_image_np * 255).astype('uint8')
# restored_image = Image.fromarray(restored_image)
# restored_image.save('restored_image.png')

##################################################

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from torchsr.models import edsr_baseline_x2

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

# Create the model, loss function, and optimizer
model = edsr_baseline_x2()  # Use the edsr_baseline_x2 model
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

# Save the trained model
torch.save(model.state_dict(), 'image_restoration_model.pth')

# Test the model on the test image
test_image_path = 'test_input_image/28279217.png'
test_image = Image.open(test_image_path).convert("RGB")
transform = transforms.Compose([transforms.ToTensor()])
test_image = transform(test_image).unsqueeze(0)  # Add batch dimension

# Load the trained model
model.load_state_dict(torch.load('image_restoration_model.pth'))
model.eval()

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
restored_image.save('restored_image_edsr.png')
