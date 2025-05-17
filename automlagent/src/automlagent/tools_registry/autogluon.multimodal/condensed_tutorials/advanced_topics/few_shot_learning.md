# Condensed: Few Shot Learning with AutoMM

Summary: This tutorial demonstrates implementing few-shot learning using AutoGluon's MultiModalPredictor for both text and image classification tasks. It provides code examples for setting up few-shot classifiers with essential configurations like problem_type="few_shot_classification", label specification, and evaluation metrics. The implementation combines foundation model features with SVM classifiers, making it particularly effective for small datasets with limited samples per class. Key functionalities include comparing few-shot vs. standard classification approaches, handling both text and image modalities, and configuring the classifier with best practices for optimal performance on small datasets.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details:

# Few Shot Learning with AutoMM

## Key Concepts
- Uses foundation model features + SVM for few shot classification
- Works for both text and image classification tasks
- Particularly effective for small datasets (few samples per class)

## Implementation Details

### Text Classification

```python
from autogluon.multimodal import MultiModalPredictor

# Few-shot classifier
predictor_fs = MultiModalPredictor(
    problem_type="few_shot_classification",
    label="label",
    eval_metric="acc"
)
predictor_fs.fit(train_df)

# Standard classifier (for comparison)
predictor_default = MultiModalPredictor(
    problem_type="classification",
    label="label",
    eval_metric="acc"
)
predictor_default.fit(train_df)
```

### Image Classification

```python
# Few-shot classifier
predictor_fs_image = MultiModalPredictor(
    problem_type="few_shot_classification",
    label="LabelName",
    eval_metric="acc"
)
predictor_fs_image.fit(train_df)

# Standard classifier (for comparison)
predictor_default_image = MultiModalPredictor(
    problem_type="classification",
    label="LabelName",
    eval_metric="acc"
)
predictor_default_image.fit(train_df)
```

## Critical Configurations
1. Set `problem_type="few_shot_classification"` for few-shot learning
2. Specify correct label column name
3. Choose appropriate evaluation metric (e.g., "acc" for accuracy)

## Best Practices
1. Use few-shot learning when you have limited training data (e.g., 8-10 samples per class)
2. Data should be in pandas DataFrame format
3. For images, ensure file paths in DataFrame are correct and accessible
4. Compare performance with standard classification to validate approach

## Important Notes
- Few-shot classification typically performs better than default classification on small datasets
- Works with both text and image modalities
- Uses foundation model features combined with SVM classifier
- For customization options, refer to the AutoMM customization documentation