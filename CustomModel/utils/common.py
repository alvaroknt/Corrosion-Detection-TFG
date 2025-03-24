# utils/common.py

from torchinfo import summary
import numpy as np
import torch
import gc
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix

def show_model_summary(model):
    """
    Prints a detailed summary of the model's layers, input/output sizes, and number of parameters.
    """
    print(summary(model, input_size=(4, 3, 512, 512), 
                  col_names=["input_size", "output_size", "num_params", "trainable"]))

def plot_confusion_matrix(y_true, y_pred):
    """
    Displays a confusion matrix for binary classification.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=['No Corrosion', 'Corrosion'], columns=['No Corrosion', 'Corrosion'])

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_df, annot=True, cmap='Blues', fmt='d')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def denormalize(image, mean, std):
    """
    Reverses normalization for an image tensor for visualization.
    """
    mean = np.array(mean)
    std = np.array(std)
    image = image.cpu().numpy().transpose(1, 2, 0)
    image = (image * std) + mean
    return np.clip(image, 0, 1)

def clean_up():
    """
    Frees up memory and clears CUDA cache to prevent OOM errors.
    """
    global model, optimizer, train_loader, test_loader
    try:
        del model
        del optimizer
        del train_loader
        del test_loader
    except NameError:
        pass
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.synchronize()
    print("Memory cleared")
