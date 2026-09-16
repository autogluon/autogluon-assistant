# Condensed: Create Dataset

Summary: This tutorial demonstrates implementing focal loss in AutoGluon's MultiModalPredictor for handling imbalanced image classification tasks. It covers creating artificially imbalanced datasets through progressive downsampling, calculating per-class weights using inverse class frequencies, and configuring focal loss hyperparameters (`optim.loss_func`, `optim.focal_loss.alpha`, `gamma`, `reduction`). Key techniques include weight normalization for class imbalance, using Swin Transformer models, and comparing focal loss vs. standard loss performance. This helps with tasks requiring imbalanced dataset handling, multi-class image classification with AutoGluon, custom loss function configuration, and class weight calculation for skewed distributions.

*This is a condensed version that preserves essential implementation details and context.*

# Focal Loss for Imbalanced Image Classification

## Dataset Setup

```python
!pip install autogluon.multimodal

from autogluon.multimodal.utils.misc import shopee_dataset
download_dir = "./ag_automm_tutorial_imgcls_focalloss"
train_data, test_data = shopee_dataset(download_dir)
```

**Create Imbalanced Dataset** (4 classes, progressively downsampled by 1/3):

```python
import numpy as np
import pandas as pd

ds = 1
imbalanced_train_data = []
for lb in range(4):
    class_data = train_data[train_data.label == lb]
    sample_index = np.random.choice(np.arange(len(class_data)), size=int(len(class_data) * ds), replace=False)
    ds /= 3
    imbalanced_train_data.append(class_data.iloc[sample_index])
imbalanced_train_data = pd.concat(imbalanced_train_data)

# Calculate per-class weights (inverse of class percentage)
weights = []
for lb in range(4):
    class_data = imbalanced_train_data[imbalanced_train_data.label == lb]
    weights.append(1 / (class_data.shape[0] / imbalanced_train_data.shape[0]))
weights = list(np.array(weights) / np.sum(weights))
```

## Training with Focal Loss

**Key Parameters:**
- `optim.loss_func`: Set to `"focal_loss"`
- `optim.focal_loss.alpha`: Per-class loss weights (list length **must** match number of classes). Use inverse of class percentage.
- `optim.focal_loss.gamma`: Focus on hard samples (higher = more focus)
- `optim.focal_loss.reduction`: `"mean"` or `"sum"`

```python
from autogluon.multimodal import MultiModalPredictor

predictor = MultiModalPredictor(label="label", problem_type="multiclass", path=model_path)

predictor.fit(
    hyperparameters={
        "model.mmdet_image.checkpoint_name": "swin_tiny_patch4_window7_224",
        "env.num_gpus": 1,
        "optim.loss_func": "focal_loss",
        "optim.focal_loss.alpha": weights,
        "optim.focal_loss.gamma": 1.0,
        "optim.focal_loss.reduction": "sum",
        "optim.max_epochs": 10,
    },
    train_data=imbalanced_train_data,
) 

predictor.evaluate(test_data, metrics=["acc"])
```

**Baseline (without Focal Loss):** Remove focal loss parameters for comparison. Focal loss significantly improves performance on imbalanced data.

**Best Practice:** Use focal loss when dealing with imbalanced datasets to improve model performance.