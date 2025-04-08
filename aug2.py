import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import kornia.augmentation as K

# Define paths
image_dir = r"C:\Users\manas\Documents\unet\train\images"
mask_dir = r"C:\Users\manas\Documents\unet\train\masks"

# Output storage paths for augmented images
aug_image_dir = r"C:\Users\SRI SAIRAM COLLEGE\Documents\UNET\aug_files\imgs"
aug_mask_dir = r"C:\Users\SRI SAIRAM COLLEGE\Documents\UNET\aug_files\labels"

# Create directories if they don’t exist
os.makedirs(aug_image_dir, exist_ok=True)
os.makedirs(aug_mask_dir, exist_ok=True)

# Define a dataset class
class UNetDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = os.listdir(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.image_filenames[idx])

        # Load image and mask
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Convert to tensor & normalize to [0,1] range
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0) / 255.0  # Ensure mask has 1 channel

        # Apply Kornia transforms if provided
        if self.transform:
            image = self.transform(image)

        return image, mask, self.image_filenames[idx]

# Kornia GPU transformations
kornia_transforms = torch.nn.Sequential(
    K.RandomHorizontalFlip(p=0.5),
    K.RandomVerticalFlip(p=0.5),
    K.RandomRotation(degrees=30),
)

# Create dataset & dataloader
train_dataset = UNetDataset(image_dir, mask_dir, transform=kornia_transforms)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Save Augmented Images
for images, masks, filenames in train_loader:
    for i in range(len(filenames)):
        # Convert tensors back to NumPy
        img_np = (images[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        mask_np = (masks[i].squeeze().cpu().numpy() * 255).astype(np.uint8)

        # Save images and masks
        cv2.imwrite(os.path.join(aug_image_dir, filenames[i]), img_np)
        cv2.imwrite(os.path.join(aug_mask_dir, filenames[i]), mask_np)

    print(f"Saved {len(filenames)} augmented images & masks.")

    break  # Run only one batch for verification
