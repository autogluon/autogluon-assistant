# Condensed: AutoMM for Semantic Segmentation - Quick Start

Summary: This tutorial demonstrates implementing semantic segmentation using AutoGluon's MultiModalPredictor with SAM (Segment Anything Model). It covers essential techniques for data preparation with path handling, zero-shot inference, and model fine-tuning using LoRA. The tutorial helps with tasks like setting up semantic segmentation pipelines, performing zero-shot evaluation, and fine-tuning SAM models for domain-specific applications. Key features include configuring the predictor for foreground-background segmentation, handling data in DataFrame format, model persistence, and performance evaluation using IoU metrics. It's particularly useful for implementing efficient semantic segmentation with pre-trained SAM models while emphasizing security considerations when loading saved models.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details:

# AutoMM for Semantic Segmentation - Quick Start

## Key Implementation Details

### Setup and Data Preparation
```python
!pip install autogluon.multimodal

# Load and prepare dataset
import pandas as pd
import os

# Expand relative paths to absolute paths
def path_expander(path, base_folder):
    path_l = path.split(';')
    return ';'.join([os.path.abspath(os.path.join(base_folder, path)) for path in path_l])

# Apply path expansion to image and label columns
for per_col in [image_col, label_col]:
    train_data[per_col] = train_data[per_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
    val_data[per_col] = val_data[per_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
    test_data[per_col] = test_data[per_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
```

### Zero-Shot Evaluation
```python
from autogluon.multimodal import MultiModalPredictor

# Initialize predictor for zero-shot inference
predictor_zero_shot = MultiModalPredictor(
    problem_type="semantic_segmentation", 
    label=label_col,
    hyperparameters={
        "model.sam.checkpoint_name": "facebook/sam-vit-base",
    },
    num_classes=1  # foreground-background segmentation
)

# Perform prediction and evaluation
pred_zero_shot = predictor_zero_shot.predict({'image': [test_data.iloc[0]['image']]})
scores = predictor_zero_shot.evaluate(test_data, metrics=["iou"])
```

### Fine-tuning SAM
```python
# Initialize predictor for fine-tuning
predictor = MultiModalPredictor(
    problem_type="semantic_segmentation", 
    label="label",
    hyperparameters={
        "model.sam.checkpoint_name": "facebook/sam-vit-base",
    },
    path=save_path,
)

# Fine-tune the model
predictor.fit(
    train_data=train_data,
    tuning_data=val_data,
    time_limit=180  # seconds
)
```

## Critical Configurations
- Problem type: "semantic_segmentation"
- Model: SAM (Segment Anything Model)
- Checkpoint: "facebook/sam-vit-base"
- Number of classes: 1 (foreground-background segmentation)
- Fine-tuning method: LoRA (Low-Rank Adaptation)

## Important Notes and Best Practices
1. Data format: DataFrame with image and label columns containing absolute paths
2. SAM without prompts outputs rough masks; fine-tuning improves domain-specific performance
3. LoRA is used for efficient fine-tuning of the large SAM model
4. **Security Warning**: Use `MultiModalPredictor.load()` only with trusted data sources due to pickle security concerns

## Model Persistence
```python
# Save is automatic after fit()
# Load saved model
loaded_predictor = MultiModalPredictor.load(save_path)
```

## Performance Evaluation
```python
# Evaluate model performance
scores = predictor.evaluate(test_data, metrics=["iou"])
```

For visualization and additional customization options, refer to the full documentation and [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm).