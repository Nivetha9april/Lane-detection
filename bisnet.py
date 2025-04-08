import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import os
from torchvision.models.segmentation import deeplabv3_resnet50

if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")

class BiSeNet(nn.Module):
    def __init__(self, num_classes=256):
        super(BiSeNet, self).__init__()
        self.backbone = deeplabv3_resnet50(pretrained=True)  # Using DeepLabV3's backbone
        self.backbone.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)  # Output classes

    def forward(self, x):
        return self.backbone(x)["out"]  # Output segmentation map

class RoadSegDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_list = os.listdir(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_list[idx])
        mask_path = os.path.join(self.mask_dir, self.image_list[idx].replace(".jpg", ".png"))

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Convert mask to grayscale

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        return image, mask.long()

transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

train_dataset = RoadSegDataset(r"C:\Users\SRI SAIRAM COLLEGE\Documents\UNET\training\images",r"C:\Users\SRI SAIRAM COLLEGE\Documents\UNET\training\labels", transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = BiSeNet(num_classes=256).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#assert mask.min() >= 0 and mask.max() < 71, f"Mask values out of range: {mask.min()} to {mask.max()}"

def train(model, dataloader, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader):.4f}")
    
    save_path = "bisenet_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

train(model, train_loader, epochs=20)

def predict(model, image_path):
    model.eval()
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Error: Cannot read image at {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    image = transform(image=image)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image).argmax(dim=1).cpu().numpy()[0]

    return output

import matplotlib.pyplot as plt

image_path = r"C:\Users\SRI SAIRAM COLLEGE\Documents\UNET\testing\images\___QXeb8e952hTD6EaQVEQ.jpg"
segmentation_map = predict(model, image_path)

    