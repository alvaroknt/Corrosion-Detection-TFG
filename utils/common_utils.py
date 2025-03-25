from torchinfo import summary

def show_model_summary(model):
    """
    Print a detailed summary of the model architecture.
    """
    print(summary(model, input_size=(2, 3, 512, 512), col_names=["input_size", "output_size", "num_params", "trainable"]))

def plot_confusion_matrix(y_true, y_pred):
    """
    Compute and display a confusion matrix for binary classification.
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt

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
    Denormalize an image tensor using the specified mean and standard deviation.
    """
    import numpy as np
    mean = np.array(mean)
    std = np.array(std)
    image = image.cpu().numpy().transpose(1, 2, 0)  # Change to (H, W, C)
    image = (image * std) + mean
    return np.clip(image, 0, 1)

def clean_up():
    """
    Clear memory and cached tensors from GPU before new training/evaluation.
    """
    import gc
    import torch
    import os
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
