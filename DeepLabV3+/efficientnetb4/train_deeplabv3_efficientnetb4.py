# DeepLabV3+/efficientnetb4/train_deeplabv3_efficientnetb4.py

import os
import time
import warnings
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, recall_score, precision_score, roc_auc_score
import segmentation_models_pytorch as smp
from carbontracker.tracker import CarbonTracker

from utils.common import show_model_summary, denormalize, plot_confusion_matrix, clean_up

warnings.filterwarnings("ignore", category=UserWarning, module="carbontracker")

# Custom dataset class for binary segmentation
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

def DL_EfficientNetB4(loss="BCE", alpha=1, gamma=2, num_epochs=10, batch_size=4,
                      shuffle_train=True, shuffle_test=False, model_name="DeepLabV3Plus_EfficientNetB4"):

    clean_up()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO: Replace with actual paths
    train_image_dir = "/path/to/train/images"
    train_mask_dir = "/path/to/train/masks"
    test_image_dir = "/path/to/test/images"
    test_mask_dir = "/path/to/test/masks"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load datasets and dataloaders
    train_dataset = CorrosionDataset(train_image_dir, train_mask_dir, transform)
    test_dataset = CorrosionDataset(test_image_dir, test_mask_dir, transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=shuffle_test)

    # Initialize DeepLabV3+ model with EfficientNetB4 backbone
    model = smp.DeepLabV3Plus(
        encoder_name="efficientnet-b4",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    ).to(device)

    show_model_summary(model)

    # Select loss function
    if loss == "BCE":
        criterion = nn.BCEWithLogitsLoss()
    elif loss == "FL":
        def focal_loss(inputs, targets, alpha=alpha, gamma=gamma):
            bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
            pt = torch.exp(-bce)
            return (alpha * (1 - pt) ** gamma * bce).mean()
        criterion = focal_loss

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    def train_model():
        model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                images, masks = images.to(device), masks.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch+1}, Training Loss: {running_loss / len(train_loader):.4f}")

    # Evaluation and visualization
    def evaluate_and_visualize():
        model.eval()
        all_preds, all_masks = [], []
        shown = 0

        with torch.no_grad():
            for images, masks in test_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                preds = torch.sigmoid(outputs) > 0.5

                all_preds.append(preds.cpu().numpy().flatten())
                all_masks.append(masks.cpu().numpy().flatten())

                if shown < 3:
                    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                    ax[0].imshow(denormalize(images[0], [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
                    ax[0].set_title("Original")
                    ax[1].imshow(masks[0].cpu().squeeze(), cmap="gray")
                    ax[1].set_title("Ground Truth")
                    ax[2].imshow(preds[0].cpu().squeeze(), cmap="gray")
                    ax[2].set_title("Prediction")
                    plt.show()
                    shown += 1

        # Compute metrics
        all_preds = np.concatenate(all_preds)
        all_masks = np.concatenate(all_masks)

        print(f"F1 Score: {f1_score(all_masks, all_preds):.4f}")
        print(f"Jaccard Score: {jaccard_score(all_masks, all_preds):.4f}")
        print(f"Accuracy: {accuracy_score(all_masks, all_preds):.4f}")
        print(f"Precision: {precision_score(all_masks, all_preds):.4f}")
        print(f"Recall: {recall_score(all_masks, all_preds):.4f}")
        print(f"Specificity: {recall_score(all_masks, all_preds, pos_label=0):.4f}")
        print(f"AUC: {roc_auc_score(all_masks, all_preds):.4f}")

        plot_confusion_matrix(all_masks, all_preds)

        # Save metrics
        metrics_file = "/content/drive/MyDrive/TFG/metricas_tfg.csv"
        metrics_data = {
            "Model": [model_name],
            "Accuracy": [accuracy_score(all_masks, all_preds)],
            "F1 Score": [f1_score(all_masks, all_preds)],
            "Jaccard Score": [jaccard_score(all_masks, all_preds)],
            "Specificity": [recall_score(all_masks, all_preds, pos_label=0)],
            "AUC": [roc_auc_score(all_masks, all_preds)],
            "Recall": [recall_score(all_masks, all_preds)],
            "Precision": [precision_score(all_masks, all_preds)]
        }
        df = pd.DataFrame(metrics_data)

        if os.path.exists(metrics_file):
            prev = pd.read_csv(metrics_file)
            df = pd.concat([prev, df], ignore_index=True)
        df.to_csv(metrics_file, index=False)

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
