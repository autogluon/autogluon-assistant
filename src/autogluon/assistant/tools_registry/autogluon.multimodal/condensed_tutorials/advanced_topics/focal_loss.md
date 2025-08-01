# Condensed: Create Dataset

Summary: This tutorial demonstrates implementing focal loss in AutoGluon for handling imbalanced image classification datasets. It covers techniques for calculating class weights, configuring focal loss parameters (alpha, gamma, reduction), and integrating them into MultiModalPredictor. The tutorial helps with training computer vision models on skewed data distributions by showing how to properly weight underrepresented classes. Key functionalities include creating imbalanced datasets, configuring focal loss hyperparameters, comparing models with and without focal loss, and using Swin Transformer architecture for image classification tasks.

*This is a condensed version that preserves essential implementation details and context.*

# Using Focal Loss for Imbalanced Data in AutoGluon

## Dataset Preparation

```python
from autogluon.multimodal.utils.misc import shopee_dataset
import numpy as np
import pandas as pd

# Download dataset
download_dir = "./ag_automm_tutorial_imgcls_focalloss"
train_data, test_data = shopee_dataset(download_dir)

# Create imbalanced dataset for demonstration
ds = 1
imbalanced_train_data = []
for lb in range(4):
    class_data = train_data[train_data.label == lb]
    sample_index = np.random.choice(np.arange(len(class_data)), size=int(len(class_data) * ds), replace=False)
    ds /= 3  # downsample 1/3 each time for each class
    imbalanced_train_data.append(class_data.iloc[sample_index])
imbalanced_train_data = pd.concat(imbalanced_train_data)

# Calculate class weights for focal loss
weights = []
for lb in range(4):
    class_data = imbalanced_train_data[imbalanced_train_data.label == lb]
    weights.append(1 / (class_data.shape[0] / imbalanced_train_data.shape[0]))
    print(f"class {lb}: num samples {len(class_data)}")
weights = list(np.array(weights) / np.sum(weights))
```

## Training with Focal Loss

```python
import uuid
from autogluon.multimodal import MultiModalPredictor

model_path = f"./tmp/{uuid.uuid4().hex}-automm_shopee_focal"

predictor = MultiModalPredictor(label="label", problem_type="multiclass", path=model_path)

predictor.fit(
    hyperparameters={
        "model.mmdet_image.checkpoint_name": "swin_tiny_patch4_window7_224",
        "env.num_gpus": 1,
        "optim.loss_func": "focal_loss",
        "optim.focal_loss.alpha": weights,  # Class weights
        "optim.focal_loss.gamma": 1.0,      # Controls focus on hard samples
        "optim.focal_loss.reduction": "sum", # How to aggregate loss
        "optim.max_epochs": 10,
    },
    train_data=imbalanced_train_data,
) 

predictor.evaluate(test_data, metrics=["acc"])
```

## Training without Focal Loss (for comparison)

```python
model_path = f"./tmp/{uuid.uuid4().hex}-automm_shopee_non_focal"

predictor2 = MultiModalPredictor(label="label", problem_type="multiclass", path=model_path)

predictor2.fit(
    hyperparameters={
        "model.mmdet_image.checkpoint_name": "swin_tiny_patch4_window7_224",
        "env.num_gpus": 1,
        "optim.max_epochs": 10,
    },
    train_data=imbalanced_train_data,
)

predictor2.evaluate(test_data, metrics=["acc"])
```

## Key Focal Loss Parameters

- **optim.loss_func**: Set to "focal_loss" to enable focal loss
- **optim.focal_loss.alpha**: List of per-class weights (must match number of classes)
- **optim.focal_loss.gamma**: Controls focus on hard samples (higher = more focus)
- **optim.focal_loss.reduction**: How to aggregate loss ("mean" or "sum")

**Best Practice**: When dealing with imbalanced datasets, focal loss can significantly improve model performance by focusing on hard-to-classify examples and properly weighting underrepresented classes.