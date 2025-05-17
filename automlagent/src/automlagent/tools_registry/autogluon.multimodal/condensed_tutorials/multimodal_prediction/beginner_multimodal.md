# Condensed: AutoMM for Image + Text + Tabular - Quick Start

Summary: This tutorial demonstrates the implementation of AutoGluon's MultiModalPredictor for handling combined image, text, and tabular data processing tasks. It provides code for essential operations including data preparation (path expansion for images), model training, prediction (standard and probability-based), evaluation, feature extraction, and model persistence. The tutorial showcases AutoGluon's automatic features like problem type detection, data modality recognition, and model selection. Key functionalities covered include multi-modal fusion, automated model training, embedding extraction, and handling various data formats. It's particularly useful for developers working on projects requiring unified processing of mixed data types with minimal manual configuration.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details:

# AutoMM for Image + Text + Tabular - Quick Start

## Key Setup
```python
!pip install autogluon.multimodal
import numpy as np
import pandas as pd
from autogluon.multimodal import MultiModalPredictor
```

## Implementation Details

### Data Preparation
1. Load multimodal data containing images, text, and tabular features
2. Process image paths:
```python
# Expand image paths to full path
def path_expander(path, base_folder):
    path_l = path.split(';')
    return ';'.join([os.path.abspath(os.path.join(base_folder, path)) for path in path_l])

# Apply to image column
train_data[image_col] = train_data[image_col].apply(
    lambda ele: path_expander(ele.split(';')[0], base_folder=dataset_path)
)
```

### Training
```python
predictor = MultiModalPredictor(label=label_col)
predictor.fit(
    train_data=train_data,
    time_limit=120  # seconds
)
```

### Key Operations
1. Prediction:
```python
# Standard prediction
predictions = predictor.predict(test_data.drop(columns=label_col))

# Probability prediction (classification only)
probas = predictor.predict_proba(test_data.drop(columns=label_col))
```

2. Evaluation:
```python
scores = predictor.evaluate(test_data, metrics=["roc_auc"])
```

3. Feature Extraction:
```python
embeddings = predictor.extract_embedding(test_data.drop(columns=label_col))
```

4. Model Persistence:
```python
# Save model
predictor.save(model_path)

# Load model
loaded_predictor = MultiModalPredictor.load(model_path)
```

## Important Notes and Best Practices

1. **Security Warning**: `MultiModalPredictor.load()` uses pickle - only load trusted model files

2. **Automatic Features**:
   - Problem type auto-detection (classification/regression)
   - Data modality detection
   - Model selection from multimodal pools
   - Late-fusion model addition for multiple backbones

3. **Data Format**: Input data must be in multimodal dataframe format with properly formatted columns for:
   - Images (paths)
   - Text
   - Tabular features

4. **Performance**: Set appropriate `time_limit` based on dataset size and requirements

This implementation supports automatic handling of:
- Image data
- Text descriptions
- Tabular features
- Multi-modal fusion
- Feature extraction
- Model persistence