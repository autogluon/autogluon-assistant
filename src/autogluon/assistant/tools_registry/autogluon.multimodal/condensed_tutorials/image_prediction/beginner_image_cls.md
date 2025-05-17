# Condensed: AutoMM for Image Classification - Quick Start

Summary: This tutorial demonstrates AutoGluon's MultiModalPredictor implementation for image classification tasks, covering essential techniques for model training, evaluation, and deployment. It provides code examples for data preparation (supporting both image paths and bytearrays), model training configuration, prediction generation, feature extraction, and model persistence. Key functionalities include accuracy evaluation, probability predictions, embedding extraction (512-2048 dimensional vectors), and flexible input handling. The tutorial is particularly useful for implementing automated image classification pipelines, with specific focus on practical aspects like model saving/loading, security considerations, and input format compatibility.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details:

# AutoMM for Image Classification - Quick Start

## Key Implementation Details

### Setup and Data Preparation
```python
from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.utils.misc import shopee_dataset

# Load dataset (supports two formats)
# 1. With image paths
train_data_path, test_data_path = shopee_dataset(download_dir)

# 2. With bytearrays
train_data_byte, test_data_byte = shopee_dataset(download_dir, is_bytearray=True)
```

### Model Training
```python
predictor = MultiModalPredictor(
    label="label",  # Column name containing target variable
    path="./model_path"  # Save directory for models
)

predictor.fit(
    train_data=train_data_path,
    time_limit=30  # Training time in seconds
)
```

### Evaluation and Prediction
```python
# Evaluate model
scores = predictor.evaluate(test_data_path, metrics=["accuracy"])

# Make predictions
predictions = predictor.predict({'image': [image_path]})
probabilities = predictor.predict_proba({'image': [image_path]})

# Extract embeddings
features = predictor.extract_embedding({'image': [image_path]})
```

### Model Persistence
```python
# Save happens automatically after fit()
# Load saved model
loaded_predictor = MultiModalPredictor.load(model_path)
```

## Critical Configurations
- Required columns: 'image' and 'label'
- Supports both image paths and bytearrays as input
- Model path must be specified for saving/loading

## Important Notes and Best Practices
1. ⚠️ Security Warning: Use `MultiModalPredictor.load()` only with trusted data sources due to pickle security risks
2. Compatible with both image paths and bytearrays for training/inference
3. Same model can be used interchangeably with paths or bytearrays
4. Default embeddings are typically 512-2048 dimensional vectors

## Supported Operations
- Image classification
- Probability prediction
- Feature extraction
- Model evaluation
- Model persistence

For advanced customization options, refer to the AutoMM customization documentation.