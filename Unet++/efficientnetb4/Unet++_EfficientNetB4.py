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
from utils.common_utils import clean_up, show_model_summary, denormalize, plot_confusion_matrix

# Suppress CarbonTracker warnings
warnings.filterwarnings("ignore", category=UserWarning, module="carbontracker")
warnings.filterwarnings("ignore", message="ElectricityMaps API key not set")
warnings.filterwarnings("ignore", message="Failed to retrieve carbon intensity")

def UNet_EfficientNetB4(loss="BCE", alpha=1, gamma=2, num_epochs=10, batch_size=4,
                         shuffle_train=True, shuffle_test=False, model_name="UNetPlusPlus_EfficientNetB4"):
    clean_up()

    # TODO: Replace with actual paths or mount your drive if in Colab
    train_image_dir = "/path/to/train/images"
    train_mask_dir = "/path/to/train/masks"
    test_image_dir = "/path/to/test/images"
    test_mask_dir = "/path/to/test/masks"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class CorrosionDataset(Dataset):
        """
        Custom dataset for loading RGB images and binary masks.
        """
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
            mask_path = os.path.join(self.mask_dir, self.masks[idx])

            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")

            mask = (np.array(mask) > 0).astype(np.float32)
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

            if self.transform:
                image = self.transform(image)

            return image, mask

    # Image transformations (normalization based on ImageNet stats)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CorrosionDataset(train_image_dir, train_mask_dir, transform=transform)
    test_dataset = CorrosionDataset(test_image_dir, test_mask_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=shuffle_test,
                             sampler=torch.utils.data.SequentialSampler(test_dataset))

    # Load UNet++ model with EfficientNetB4 encoder
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b4",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    )
    model = model.to(device)
    show_model_summary(model)

    # Define loss function
    if loss == "BCE":
        criterion = nn.BCEWithLogitsLoss()
    elif loss == "FL":
        def focal_loss(inputs, targets, alpha=alpha, gamma=gamma):
            bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
            pt = torch.exp(-bce)
            return (alpha * (1 - pt) ** gamma * bce).mean()
        criterion = focal_loss

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

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
            avg_loss = running_loss / len(train_loader)
            print(f"Epoch {epoch+1} - Training Loss: {avg_loss:.4f}")

    def evaluate_model():
        model.eval()
        all_preds = []
        all_masks = []
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
                    ax[0].imshow(denormalize(images[0], [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
                    ax[0].set_title("Original Image")
                    ax[1].imshow(masks[0].cpu().squeeze(), cmap="gray")
                    ax[1].set_title("Ground Truth")
                    ax[2].imshow(preds[0].cpu().squeeze(), cmap="gray")
                    ax[2].set_title("Prediction")
                    plt.show()
                    images_shown += 1

        all_preds = np.concatenate(all_preds)
        all_masks = np.concatenate(all_masks)

        acc = accuracy_score(all_masks, all_preds)
        f1 = f1_score(all_masks, all_preds)
        jaccard = jaccard_score(all_masks, all_preds)
        recall = recall_score(all_masks, all_preds)
        precision = precision_score(all_masks, all_preds)
        specificity = recall_score(all_masks, all_preds, pos_label=0)
        auc = roc_auc_score(all_masks, all_preds)

        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Jaccard Index: {jaccard:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print(f"AUC: {auc:.4f}")

        plot_confusion_matrix(all_masks, all_preds)

    tracker = CarbonTracker(epochs=2)
    tracker.epoch_start()
    train_model()
    tracker.epoch_end()

    tracker.epoch_start()
    evaluate_model()
    tracker.epoch_end()
