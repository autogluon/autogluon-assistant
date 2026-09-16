# Condensed: ```python

Summary: This tutorial demonstrates AutoGluon's MultiModalPredictor for automated multimodal machine learning combining images, text, and tabular data. It covers critical data preparation techniques including formatting image columns as file path strings, converting relative to absolute paths, and handling multiple images per record. The implementation shows a complete workflow: installing AutoGluon, loading datasets with pandas, preprocessing multimodal features, training with time-limited AutoML (automatic problem type inference, modality detection, model selection, and late-fusion), generating predictions and probabilities, and evaluating with custom metrics. Useful for building classification/regression models on datasets with mixed data types (images, text, structured data) with minimal code and automatic feature engineering.

*This is a condensed version that preserves essential implementation details and context.*

# AutoGluon MultiModalPredictor Quick Start

## Installation & Setup

```python
!python -m pip install --upgrade pip
!python -m pip install autogluon

import os
import warnings
import numpy as np

warnings.filterwarnings('ignore')
np.random.seed(123)
```

## Data Preparation

**Dataset**: PetFinder adoption prediction (binary classification: 0=slow, 1=fast adoption)

```python
from autogluon.core.utils.loaders import load_zip
import pandas as pd

# Download and load data
download_dir = './ag_multimodal_tutorial'
zip_file = 'https://automl-mm-bench.s3.amazonaws.com/petfinder_for_tutorial.zip'
load_zip.unzip(zip_file, unzip_dir=download_dir)

dataset_path = f'{download_dir}/petfinder_for_tutorial'
train_data = pd.read_csv(f'{dataset_path}/train.csv', index_col=0)
test_data = pd.read_csv(f'{dataset_path}/test.csv', index_col=0)

label_col = 'AdoptionSpeed'
```

### Image Column Formatting

**Critical**: Image columns must contain strings with paths to single image files.

```python
image_col = 'Images'

# Extract first image only
train_data[image_col] = train_data[image_col].apply(lambda ele: ele.split(';')[0])
test_data[image_col] = test_data[image_col].apply(lambda ele: ele.split(';')[0])

# Convert to absolute paths
def path_expander(path, base_folder):
    path_l = path.split(';')
    return ';'.join([os.path.abspath(os.path.join(base_folder, path)) for path in path_l])

train_data[image_col] = train_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
test_data[image_col] = test_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
```

## Training

```python
from autogluon.multimodal import MultiModalPredictor

predictor = MultiModalPredictor(label=label_col).fit(
    train_data=train_data,
    time_limit=120  # seconds; increase for better performance
)
```

**Key Features**: Automatically handles problem type inference, feature modality detection, model selection, and late-fusion (MLP/transformer) for multiple backbones.

## Inference & Evaluation

```python
# Predictions
predictions = predictor.predict(test_data.drop(columns=label_col))

# Prediction probabilities (classification)
probs = predictor.predict_proba(test_data.drop(columns=label_col))

# Evaluation
scores = predictor.evaluate(test_data, metrics=["roc_auc"])
```

**Note**: Dataset includes images, text descriptions, and tabular features (age, breed, color, etc.) - all handled automatically by MultiModalPredictor.