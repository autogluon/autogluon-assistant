# Condensed: ```python

Summary: This tutorial demonstrates image classification using AutoGluon MultiModal, covering implementation of a complete workflow from setup to deployment. It teaches how to: load image data (both file paths and bytearrays), train classification models with minimal code using MultiModalPredictor, evaluate model performance, make predictions on new images, extract feature embeddings for transfer learning, and save/load models. Key features include time-constrained training, handling multiple input formats, probability-based predictions, and feature extraction capabilities - all with AutoGluon's simplified API that abstracts away complex deep learning implementation details.

*This is a condensed version that preserves essential implementation details and context.*

# Image Classification with AutoGluon MultiModal

## Setup and Data Loading

```python
!pip install autogluon.multimodal

import warnings
warnings.filterwarnings('ignore')
import pandas as pd

from autogluon.multimodal.utils.misc import shopee_dataset
download_dir = './ag_automm_tutorial_imgcls'
train_data_path, test_data_path = shopee_dataset(download_dir)
```

AutoGluon supports both image paths and bytearrays:

```python
# Load dataset with bytearrays
train_data_byte, test_data_byte = shopee_dataset(download_dir, is_bytearray=True)
```

## Training a Model

```python
from autogluon.multimodal import MultiModalPredictor
import uuid
model_path = f"./tmp/{uuid.uuid4().hex}-automm_shopee"
predictor = MultiModalPredictor(label="label", path=model_path)
predictor.fit(
    train_data=train_data_path,
    time_limit=30, # seconds
)
```

Key parameters:
- `label`: Column name containing target variable
- `path`: Directory for saving models and outputs
- `time_limit`: Training time in seconds

## Evaluation

```python
# Evaluate with image paths
scores = predictor.evaluate(test_data_path, metrics=["accuracy"])
print('Top-1 test acc: %.3f' % scores["accuracy"])

# Evaluate with bytearrays
scores = predictor.evaluate(test_data_byte, metrics=["accuracy"])
print('Top-1 test acc: %.3f' % scores["accuracy"])
```

## Prediction

```python
# Predict label for a single image
image_path = test_data_path.iloc[0]['image']
predictions = predictor.predict({'image': [image_path]})

# Get class probabilities
proba = predictor.predict_proba({'image': [image_path]})

# Works with bytearrays too
image_byte = test_data_byte.iloc[0]['image']
predictions = predictor.predict({'image': [image_byte]})
proba = predictor.predict_proba({'image': [image_byte]})
```

## Feature Extraction

```python
# Extract embeddings from images
feature = predictor.extract_embedding({'image': [image_path]})
print(feature[0].shape)  # N-dimensional feature vector (typically 512-2048)

# Works with bytearrays too
feature = predictor.extract_embedding({'image': [image_byte]})
```

## Save and Load

```python
# Model is automatically saved during fit()
loaded_predictor = MultiModalPredictor.load(model_path)
load_proba = loaded_predictor.predict_proba({'image': [image_path]})
```

⚠️ **Warning**: `MultiModalPredictor.load()` uses `pickle` module which can be insecure. Only load models from trusted sources.

For customization options, refer to the "Customize AutoMM" documentation.