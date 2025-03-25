# CustomModel/resnet50/train_backbone_resnet50.py

import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from torchinfo import summary
from carbontracker.tracker import CarbonTracker
from PIL import Image
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from utils.common_utils import show_model_summary, plot_confusion_matrix, denormalize, clean_up

from sklearn.metrics import (
    accuracy_score, f1_score, jaccard_score, precision_score,
    recall_score, roc_auc_score
)

warnings.filterwarnings("ignore", category=UserWarning, module="carbontracker")

# Dataset class for binary segmentation of corrosion
class CorrosionDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load RGB image
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")

        # Load grayscale mask and convert to binary tensor
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        mask = Image.open(mask_path).convert("L")
        mask = (np.array(mask) > 0).astype(np.float32)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        if self.transform:
            image = self.transform(image)

        return image, mask

# ResNet50-based binary segmentation model
class ResNet50BinaryClassifier(nn.Module):
    def __init__(self):
        super(ResNet50BinaryClassifier, self).__init__()
        weights = ResNet50_Weights.DEFAULT
        self.backbone = resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        # Decoder with progressive upsampling
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(1024, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 1, kernel_size=3, padding=1),
            nn.Upsample(size=(512, 512), mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.upsample(x)
        return x

def BB_ResNet50(loss="BCE", alpha=1, gamma=2, num_epochs=10, batch_size=4,
                shuffle_train=True, shuffle_test=False, model_name="BB_ResNet50"):

    clean_up()

    # TODO: Replace with actual dataset paths
    train_image_dir = "/path/to/train/images"
    train_mask_dir = "/path/to/train/masks"
    test_image_dir = "/path/to/test/images"
    test_mask_dir = "/path/to/test/masks"

    # ImageNet normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_dataset = CorrosionDataset(train_image_dir, train_mask_dir, transform=transform)
    test_dataset = CorrosionDataset(test_image_dir, test_mask_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle_test)

    # Model, optimizer, and loss setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet50BinaryClassifier().to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    if loss == "BCE":
        criterion = nn.BCEWithLogitsLoss()
    elif loss == "FL":
        def focal_loss(inputs, targets, alpha=alpha, gamma=gamma):
            bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
            pt = torch.exp(-bce_loss)
            return (alpha * (1 - pt) ** gamma * bce_loss).mean()
        criterion = focal_loss

    show_model_summary(model)

    def train_model():
        scaler = GradScaler()
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0

            for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                images, masks = images.to(device), masks.to(device)
                optimizer.zero_grad()

                # Forward and backward pass with mixed precision
                with autocast():
                    outputs = model(images)
                    outputs = F.interpolate(outputs, size=(512, 512), mode="bilinear", align_corners=False)
                    loss = criterion(outputs, masks)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()

            print(f"Epoch {epoch+1}, Training Loss: {train_loss / len(train_loader):.4f}")

    def evaluate_and_visualize():
        model.eval()
        all_preds, all_masks = [], []
        images_shown = 0

        with torch.no_grad():
            for images, masks in test_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                outputs = F.interpolate(outputs, size=(512, 512), mode="bilinear", align_corners=False)
                preds = (torch.sigmoid(outputs) > 0.5).float()

                all_preds.append(preds.cpu().numpy().flatten())
                all_masks.append(masks.cpu().numpy().flatten())

                if images_shown < 3:
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                    ax[0].imshow(denormalize(images[0], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
                    ax[0].set_title("Original Image")
                    ax[1].imshow(masks[0].cpu().squeeze().numpy(), cmap="gray")
                    ax[1].set_title("Ground Truth")
                    ax[2].imshow(preds[0].cpu().squeeze().numpy(), cmap="gray")
                    ax[2].set_title("Prediction")
                    plt.show()
                    images_shown += 1

        # Metrics
        all_preds = np.concatenate(all_preds)
        all_masks = np.concatenate(all_masks)

        print(f"F1 Score: {f1_score(all_masks, all_preds):.4f}")
        print(f"Jaccard Index: {jaccard_score(all_masks, all_preds):.4f}")
        print(f"Accuracy: {accuracy_score(all_masks, all_preds):.4f}")
        print(f"Precision: {precision_score(all_masks, all_preds):.4f}")
        print(f"Recall: {recall_score(all_masks, all_preds):.4f}")
        print(f"Specificity: {recall_score(all_masks, all_preds, pos_label=0):.4f}")
        print(f"AUC: {roc_auc_score(all_masks, all_preds):.4f}")

        plot_confusion_matrix(all_masks, all_preds)

    tracker = CarbonTracker(epochs=2)
    t0 = time.time()
    tracker.epoch_start()
    train_model()
    tracker.epoch_end()
    print(f"Training Time: {time.time() - t0:.2f}s")

    t0 = time.time()
    tracker.epoch_start()
    evaluate_and_visualize()
    tracker.epoch_end()
    print(f"Evaluation Time: {time.time() - t0:.2f}s")
