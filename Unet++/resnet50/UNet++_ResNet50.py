# UNetPlusPlus_ResNet50.py

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, recall_score, precision_score, roc_auc_score
import segmentation_models_pytorch as smp
from torch import nn
from torch.nn import functional as F
from carbontracker.tracker import CarbonTracker
import warnings

from utils.common_utils import show_model_summary, plot_confusion_matrix, denormalize, clean_up

# Suppress specific warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="carbontracker")
warnings.filterwarnings("ignore", message="ElectricityMaps API key not set")
warnings.filterwarnings("ignore", message="Failed to retrieve carbon intensity")

def UNet_ResNet50_sincong(loss="BCE", alpha=1, gamma=2, num_epochs=10, batch_size=4, shuffle_train=True, shuffle_test=False, model_name="UNetPlusPlus_ResNet50"):
    clean_up()

    # TODO: Replace with actual paths or mount your drive if in Colab
    train_image_dir = "/path/to/train/images"
    train_mask_dir = "/path/to/train/masks"
    test_image_dir = "/path/to/test/images"
    test_mask_dir = "/path/to/test/masks"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Custom dataset class for loading RGB images and binary masks
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
            img_path = os.path.join(self.image_dir, self.images[idx])
            image = Image.open(img_path).convert("RGB")

            mask_path = os.path.join(self.mask_dir, self.masks[idx])
            mask = Image.open(mask_path).convert("L")

            mask = (np.array(mask) > 0).astype(np.float32)
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

            if self.transform:
                image = self.transform(image)

            return image, mask

    # Transformations for input normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Initialize datasets and loaders
    train_dataset = CorrosionDataset(train_image_dir, train_mask_dir, transform=transform)
    test_dataset = CorrosionDataset(test_image_dir, test_mask_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=shuffle_test, sampler=torch.utils.data.SequentialSampler(test_dataset))

    # Define the model with ResNet50 backbone
    model = smp.UnetPlusPlus(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    )
    model = model.to(device)

    # Display model summary
    show_model_summary(model)

    # Define loss function
    if loss == "BCE":
        criterion = nn.BCEWithLogitsLoss()
    elif loss == "FL":
        def focal_loss(inputs, targets, alpha=alpha, gamma=gamma):
            bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
            pt = torch.exp(-bce_loss)
            focal = alpha * (1 - pt) ** gamma * bce_loss
            return focal.mean()
        criterion = focal_loss

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    def train_model(model, train_loader, criterion, optimizer, num_epochs):
        model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
            for images, masks in pbar:
                images, masks = images.to(device), masks.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch+1}, Training Loss: {running_loss / len(train_loader):.4f}")

    # Evaluation loop
    def evaluate_and_visualize(model, test_loader):
        model.eval()
        all_preds, all_masks = [], []
        images_shown = 0

        with torch.no_grad():
            for images, masks in test_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                preds = torch.sigmoid(outputs) > 0.5

                all_preds.append(preds.cpu().numpy().flatten())
                all_masks.append(masks.cpu().numpy().flatten())

                if images_shown < 3:
                    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                    ax[0].imshow(denormalize(images.squeeze(0), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
                    ax[0].set_title("Original Image")
                    ax[1].imshow(masks.squeeze().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
                    ax[1].set_title("Ground Truth Mask")
                    ax[2].imshow(preds.squeeze().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
                    ax[2].set_title("Prediction")
                    plt.show()
                    images_shown += 1

        # Concatenate predictions for metric computation
        all_preds = np.concatenate(all_preds)
        all_masks = np.concatenate(all_masks)

        # Compute evaluation metrics
        print(f"F1 Score: {f1_score(all_masks, all_preds, average='binary'):.4f}")
        print(f"Jaccard Score: {jaccard_score(all_masks, all_preds, average='binary'):.4f}")
        print(f"Accuracy: {accuracy_score(all_masks, all_preds):.4f}")
        print(f"Precision: {precision_score(all_masks, all_preds):.4f}")
        print(f"Recall: {recall_score(all_masks, all_preds):.4f}")
        print(f"Specificity: {recall_score(all_masks, all_preds, pos_label=0):.4f}")
        print(f"AUC: {roc_auc_score(all_masks, all_preds):.4f}")

        # Display confusion matrix
        plot_confusion_matrix(all_masks, all_preds)

    # Run training and evaluation with CarbonTracker for monitoring
    tracker = CarbonTracker(epochs=2)
    tracker.epoch_start()
    train_model(model, train_loader, criterion, optimizer, num_epochs)
    tracker.epoch_end()

    tracker.epoch_start()
    evaluate_and_visualize(model, test_loader)
    tracker.epoch_end()
