import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import scipy.io as sio
import numpy as np

# Custom Dataset for loading images and ground truth from .mat
class PeopleCountingDataset(Dataset):
    def __init__(self, image_dir, mat_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        mat = sio.loadmat(mat_file)
        self.annotations = mat["image_info"][0]
        self.image_files = [entry[0][0][0] for entry in self.annotations]
        self.counts = [entry[0][0][1][0][0] for entry in self.annotations]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.counts[idx], dtype=torch.float32)
        return image, label

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load dataset
# NOTE: Update these paths to match your local setup
image_dir = "/path/to/ShanghaiTech/part_A_final/train_data/images"
mat_file = "/path/to/ShanghaiTech/part_A_final/train_data/ground_truth/GT_IMG.mat"
dataset = PeopleCountingDataset(image_dir, mat_file, transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Load pretrained model and modify for regression
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 1)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop (example with 1 epoch)
for epoch in range(1):
    model.train()
    total_loss = 0
    for images, targets in dataloader:
        preds = model(images).squeeze()
        loss = criterion(preds, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}], Loss: {total_loss:.4f}")
