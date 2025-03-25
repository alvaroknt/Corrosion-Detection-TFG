# UNetPlusPlus_ResNet50.py

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, recall_score, precision_score, roc_auc_score
import segmentation_models_pytorch as smp
from torch import nn
from carbontracker.tracker import CarbonTracker
import warnings

from utils.common_utils import show_model_summary, plot_confusion_matrix, denormalize, clean_up

warnings.filterwarnings("ignore", category=UserWarning, module="carbontracker")
warnings.filterwarnings("ignore", message="ElectricityMaps API key not set")
warnings.filterwarnings("ignore", message="Failed to retrieve carbon intensity")

def UNet_ResNet50_sincong(loss="BCE", alpha=1, gamma=2, num_epochs=10, batch_size=4, shuffle_train=True, shuffle_test=False, model_name="UNetPlusPlus_ResNet50"):
    clean_up()

    # Define dataset directories
    train_image_dir = "/content/drive/MyDrive/TFG/Imagenes/DataAugm/Train/images_train_aug"
    train_mask_dir = "/content/drive/MyDrive/TFG/Imagenes/DataAugm/Train/masks_bin_train_aug"
    test_image_dir = "/content/drive/MyDrive/TFG/Imagenes/512x512/Test/images_test_512"
    test_mask_dir = "/content/drive/MyDrive/TFG/Imagenes/512x512/Test/mask_bin_test"

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
            image = Image.open(os.path.join(self.image_dir, self.images[idx])).convert("RGB")
            mask = Image.open(os.path.join(self.mask_dir, self.masks[idx])).convert("L")
            mask = (np.array(mask) > 0).astype(np.float32)
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
            if self.transform:
                image = self.transform(image)
            return image, mask

    # Image transformations (standard ImageNet normalization)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CorrosionDataset(train_image_dir, train_mask_dir, transform=transform)
    test_dataset = CorrosionDataset(test_image_dir, test_mask_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=shuffle_test)

    # Initialize UNet++ model with ResNet50 encoder
    model = smp.UnetPlusPlus(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    ).to(device)

    show_model_summary(model)

    # Define loss function
    if loss == "BCE":
        criterion = nn.BCEWithLogitsLoss()
    elif loss == "FL":
        def focal_loss(inputs, targets, alpha=alpha, gamma=gamma):
            bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
            pt = torch.exp(-bce_loss)
            return (alpha * (1 - pt) ** gamma * bce_loss).mean()
        criterion = focal_loss

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    def train_model():
        model.train()
        for epoch in range(num_epochs):
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
            running_loss = 0.0
            for images, masks in pbar:
                images, masks = images.to(device), masks.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss_val = criterion(outputs, masks)
                loss_val.backward()
                optimizer.step()
                running_loss += loss_val.item()
            print(f"Epoch {epoch+1} Training Loss: {running_loss / len(train_loader):.4f}")

    # Evaluation and metrics
    def evaluate_and_visualize():
        model.eval()
        images_shown = 0
        all_preds, all_masks = [], []
        with torch.no_grad():
            for images, masks in test_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                preds = (torch.sigmoid(outputs) > 0.5)
                all_preds.append(preds.cpu().numpy().flatten())
                all_masks.append(masks.cpu().numpy().flatten())

                if images_shown < 3:
                    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                    ax[0].imshow(denormalize(images.squeeze(0), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
                    ax[0].set_title("Original Image")
                    ax[1].imshow(masks.squeeze().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
                    ax[1].set_title("Ground Truth")
                    ax[2].imshow(preds.squeeze().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
                    ax[2].set_title("Prediction")
                    plt.show()
                    images_shown += 1

        all_preds = np.concatenate(all_preds)
        all_masks = np.concatenate(all_masks)

        accuracy = accuracy_score(all_masks, all_preds)
        f1 = f1_score(all_masks, all_preds)
        jaccard = jaccard_score(all_masks, all_preds)
        specificity = recall_score(all_masks, all_preds, pos_label=0)
        recall = recall_score(all_masks, all_preds)
        precision = precision_score(all_masks, all_preds)
        auc = roc_auc_score(all_masks, all_preds)

        print(f"F1 Score: {f1:.4f}\nJaccard Score: {jaccard:.4f}\nAccuracy: {accuracy:.4f}\n"
              f"Precision: {precision:.4f}\nRecall: {recall:.4f}\nSpecificity: {specificity:.4f}\nAUC: {auc:.4f}")

        metrics_file = "/content/drive/MyDrive/TFG/metricas_tfg.csv"
        metrics_df = pd.DataFrame({
            "Model": [model_name],
            "Accuracy": [accuracy],
            "F1 Score": [f1],
            "Jaccard Score": [jaccard],
            "Specificity": [specificity],
            "AUC": [auc],
            "Recall": [recall],
            "Precision": [precision],
        })

        if os.path.exists(metrics_file):
            existing = pd.read_csv(metrics_file)
            updated = pd.concat([existing, metrics_df], ignore_index=True)
            updated.to_csv(metrics_file, index=False)
        else:
            metrics_df.to_csv(metrics_file, index=False)

        plot_confusion_matrix(all_masks, all_preds)

    # Train and evaluate with CarbonTracker
    tracker = CarbonTracker(epochs=2)
    t0 = time.time(); tracker.epoch_start()
    train_model()
    tracker.epoch_end(); print(f"Training time: {time.time() - t0:.2f}s")

    t0 = time.time(); tracker.epoch_start()
    evaluate_and_visualize()
    tracker.epoch_end(); print(f"Evaluation time: {time.time() - t0:.2f}s")
