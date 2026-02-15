# TexVision-Pro Architecture

This document outlines the high-level architecture of the TexVision-Pro system.

## 1. System Overview

TexVision-Pro is a modular deep learning system designed for texture classification and analysis. It follows a standard pipeline approach:
**Data Ingestion -> Preprocessing -> Model Inference -> Post-processing/Explainability**

## 2. Directory Structure & Components

### 2.1 Configuration (`configs/`)
The system is data-driven, controlled by YAML configuration files.
- `datasets.yaml`: Defines dataset paths, splits, and specific transformations.
- `model.yaml`: Specifies the backbone architecture (e.g., ResNet50), pretrained weights usage, and head configuration.
- `training.yaml`: Contains training hyperparameters (LR, batch size, optimizer) and logging settings.

### 2.2 Data Pipeline (`torch.utils.data`)
- **Dataset**: Custom wrappers for DTD and KTH-TIPS that inherit from `torch.utils.data.Dataset`.
- **Transforms**: `torchvision.transforms` or `albumentations` are used for:
    - Resizing (Standardizing input size)
    - Normalization (ImageNet statistics)
    - Augmentation (Random crops, flips during training)
- **DataLoader**: Batching and shuffling with multi-worker support.

### 2.3 Model Architecture (`torch.nn.Module`)
The core model is a composition of:
1.  **Backbone**: A feature extractor (CNN or Transformer). Supported backbones:
    - ResNet (18, 50, 101)
    - EfficientNet
    - Vision Transformer (ViT)
2.  **Pooling**: Global Average Pooling (GAP) to reduce spatial dimensions.
3.  **Classifier Head**: A flexible head (Linear or MLP) mapping features to class logits.

### 2.4 Training Loop (`training/`)
- **Loss Function**: CrossEntropyLoss (standard), Focal Loss (class imbalance), etc.
- **Optimizer**: AdamW or SGD with momentum.
- **Scheduler**: Cosine Annealing or StepLR for learning rate decay.
- **Checkpointing**: Saves model state based on validation metrics.

### 2.5 Evaluation & Explainability (`evaluation/`)
- **Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix.
- **Grad-CAM**: Generates heatmaps by computing gradients of the target class score with respect to the final convolutional layer feature map.
- **Uncertainty**: Uses Monte Carlo Dropout (running inference multiple times with dropout enabled) to estimate predictive uncertainty.

## 3. Data Flow

1.  **Input**: Raw Image (RGB).
2.  **Preprocessing**: Resize -> Normalize -> Tensor.
3.  **Forward Pass**:
    - `x = Backbone(input)`
    - `features = Pooling(x)`
    - `logits = Head(features)`
4.  **Backward Pass (Training)**:
    - Compute Loss(logits, targets)
    - Backpropagate gradients
    - Update weights
5.  **Inference/Explainability**:
    - For Grad-CAM: Retain gradients of feature maps.
    - For Uncertainty: Run N forward passes, compute mean and variance of logits.

## 4. Experiment Tracking
Experiments are logged using TensorBoard (scalars, images) and a local `experiment_log.md`.
