# 🧠 Corrosion Detection with Deep Learning

This repository contains the code, models, and results of my Final Degree Project (Trabajo de Fin de Grado), focused on the detection of corrosion using deep learning techniques.

---

## 🔍 Objective

The aim of this project is to develop and compare several deep learning models for image segmentation, capable of detecting corrosion from labeled images.

---

## 🗂️ Repository Structure

- `images/` – Input images used for training and testing  
- `masks/` – Ground truth segmentation masks  
- `models/` – Scripts for model architectures and training  
- `results/` – Evaluation metrics and visual predictions  
- `notebooks/` – Jupyter notebooks used during the analysis  
- `utils/` – Helper functions (data loaders, metrics, etc.)

---

## 🧪 Models Used

- U-Net with MobileNetV2 backbone  
- DeepLabV3+ with ResNet50 and EfficientNetB0  
- Custom binary classification architectures  

---

## 📁 Dataset

The dataset used in this project is based on [corrosion_cs_classification](https://github.com/beric7/corrosion_cs_classification), which contains images of steel corrosion with corresponding segmentation masks in 3 classes.

For this project, the dataset was adapted to a **binary classification task**, where:
- `0` = Non-corroded
- `1` = Corroded (merging all original corrosion levels)

> ⚠️ The dataset is not included in this repository due to size and licensing constraints.  
> Please refer to the [original repository](https://github.com/beric7/corrosion_cs_classification) to download the data.

A sample of the transformation process is shown below:

| Original Image | Original Mask | Binary Mask |
|----------------|----------------|--------------|
| ![original](images/example_original.jpg) | ![original_mask](images/example_mask_original.png) | ![binary](images/example_mask_binary.png) |

---

## ▶️ How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/corrosion-detection.git
   cd corrosion-detection
