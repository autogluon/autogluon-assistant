# Condensed: Handling Class Imbalance with AutoMM - Focal Loss

Summary: This tutorial demonstrates implementing Focal Loss in AutoGluon's MultiModalPredictor to handle class imbalance problems. It provides specific code for configuring Focal Loss parameters (alpha, gamma, reduction) and calculating class weights from imbalanced data distributions. The tutorial helps with tasks involving imbalanced classification, particularly multiclass problems, by showing how to properly weight classes and focus on hard samples. Key features covered include custom loss function configuration, class weight calculation, and integration with AutoGluon's MultiModalPredictor, making it valuable for developers working on imbalanced dataset challenges.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details:

# Handling Class Imbalance with AutoMM - Focal Loss

## Key Implementation Details

### Setup
```python
from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.utils.misc import shopee_dataset
```

### Dataset Preparation
1. Load shopee dataset (4 classes, 200 samples each in training)
2. Create imbalanced distribution by downsampling:
```python
# Calculate class weights for imbalanced data
weights = []
for lb in range(4):
    class_data = imbalanced_train_data[imbalanced_train_data.label == lb]
    weights.append(1 / (class_data.shape[0] / imbalanced_train_data.shape[0]))
weights = list(np.array(weights) / np.sum(weights))
```

### Focal Loss Implementation
```python
predictor = MultiModalPredictor(label="label", problem_type="multiclass", path=model_path)

# Configure Focal Loss
predictor.fit(
    hyperparameters={
        "model.mmdet_image.checkpoint_name": "swin_tiny_patch4_window7_224",
        "env.num_gpus": 1,
        "optimization.loss_function": "focal_loss",
        "optimization.focal_loss.alpha": weights,  # Class weights
        "optimization.focal_loss.gamma": 1.0,      # Focus parameter
        "optimization.focal_loss.reduction": "sum", # Loss aggregation
        "optimization.max_epochs": 10,
    },
    train_data=imbalanced_train_data,
)
```

## Critical Configurations

### Focal Loss Parameters
- `optimization.focal_loss.alpha`: List of per-class weights (must match number of classes)
- `optimization.focal_loss.gamma`: Controls focus on hard samples (higher = more focus)
- `optimization.focal_loss.reduction`: Aggregation method ("mean" or "sum")

## Best Practices
1. Use inverse of class sample percentages for `alpha` weights
2. Consider focal loss when dealing with imbalanced datasets
3. Experiment with different `gamma` values for optimal performance

## Important Notes
- Focal loss helps balance both:
  - Hard/easy samples
  - Uneven class distribution
- Length of `alpha` list must match total number of classes
- Performance typically improves on imbalanced datasets compared to standard loss functions