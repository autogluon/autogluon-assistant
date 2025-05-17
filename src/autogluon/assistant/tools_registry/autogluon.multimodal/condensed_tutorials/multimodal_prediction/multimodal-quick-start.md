# Condensed: AutoGluon Multimodal - Quick Start

Summary: This tutorial demonstrates the implementation of AutoGluon's MultiModalPredictor for multimodal machine learning tasks, specifically focusing on combining image, text, and tabular data for classification problems. It provides code for data preparation (handling image paths and DataFrame formatting), model initialization, training, and prediction workflows. Key functionalities covered include automatic problem type detection, feature modality detection, model selection, and late-fusion model addition. The tutorial is particularly useful for tasks requiring unified processing of multiple data types, offering implementation details for prediction, probability scoring, and performance evaluation, while also highlighting advanced features like embedding extraction and model distillation.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details:

# AutoGluon MultiModalPredictor Quick Start

## Key Setup
```python
# Install and import
!python -m pip install autogluon

import numpy as np
import pandas as pd
from autogluon.multimodal import MultiModalPredictor
```

## Implementation Details

### 1. Data Preparation
- Dataset: PetFinder (simplified version for adoption speed prediction)
- Features: Images, text descriptions, and tabular data
- Target: Binary classification (AdoptionSpeed: 0=slow, 1=fast)

```python
# Load data
train_data = pd.read_csv('train.csv', index_col=0)
test_data = pd.read_csv('test.csv', index_col=0)
label_col = 'AdoptionSpeed'

# Image path handling
image_col = 'Images'
# Convert multiple image paths to single image path
train_data[image_col] = train_data[image_col].apply(lambda ele: ele.split(';')[0])
test_data[image_col] = test_data[image_col].apply(lambda ele: ele.split(';')[0])

# Expand relative paths to absolute paths
def path_expander(path, base_folder):
    path_l = path.split(';')
    return ';'.join([os.path.abspath(os.path.join(base_folder, path)) for path in path_l])
```

### 2. Model Training
```python
# Initialize and train
predictor = MultiModalPredictor(label=label_col).fit(
    train_data=train_data,
    time_limit=120  # Adjust time limit based on needs
)
```

### 3. Prediction
```python
# Make predictions
predictions = predictor.predict(test_data.drop(columns=label_col))

# Get probability scores
probs = predictor.predict_proba(test_data.drop(columns=label_col))

# Evaluate performance
scores = predictor.evaluate(test_data, metrics=["roc_auc"])
```

## Important Notes

1. **Data Format Requirements**:
   - Image columns must contain paths to single image files
   - Data should be in pandas DataFrame format
   - Label column must be specified

2. **Automatic Features**:
   - Problem type detection (classification/regression)
   - Feature modality detection
   - Model selection from multimodal pools
   - Late-fusion model addition for multiple backbones

3. **Best Practices**:
   - Increase `time_limit` for better model performance
   - Ensure image paths are correctly formatted
   - Verify all image files exist at specified paths

4. **Advanced Features Available**:
   - Embedding extraction
   - Model distillation
   - Fine-tuning
   - Text/image prediction
   - Semantic matching

This implementation provides a foundation for multimodal prediction tasks combining images, text, and tabular data using AutoGluon's MultiModalPredictor.