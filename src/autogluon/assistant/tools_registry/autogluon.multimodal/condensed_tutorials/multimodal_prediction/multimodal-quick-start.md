# Condensed: ```python

Summary: This tutorial demonstrates AutoGluon MultiModal for multimodal machine learning, focusing on image-text classification with the PetFinder dataset. It covers implementation techniques for preparing multimodal data (handling image paths and text features), training a MultiModalPredictor with automatic modality detection and model selection, and making predictions. Key functionalities include automatic problem type inference, multimodal feature handling, time-constrained training, classification prediction, probability estimation, and performance evaluation using metrics like ROC-AUC. This knowledge helps with tasks requiring combined processing of images and tabular data for classification problems.

*This is a condensed version that preserves essential implementation details and context.*

# AutoGluon MultiModal Quickstart Tutorial

## Setup

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

Download and prepare the PetFinder dataset (simplified version with binary adoption speed):

```python
from autogluon.core.utils.loaders import load_zip
import pandas as pd

# Download dataset
download_dir = './ag_multimodal_tutorial'
zip_file = 'https://automl-mm-bench.s3.amazonaws.com/petfinder_for_tutorial.zip'
load_zip.unzip(zip_file, unzip_dir=download_dir)

# Load data
dataset_path = f'{download_dir}/petfinder_for_tutorial'
train_data = pd.read_csv(f'{dataset_path}/train.csv', index_col=0)
test_data = pd.read_csv(f'{dataset_path}/test.csv', index_col=0)
label_col = 'AdoptionSpeed'

# Process image paths (take only first image per record)
image_col = 'Images'
train_data[image_col] = train_data[image_col].apply(lambda ele: ele.split(';')[0])
test_data[image_col] = test_data[image_col].apply(lambda ele: ele.split(';')[0])

# Expand image paths to absolute paths
def path_expander(path, base_folder):
    path_l = path.split(';')
    return ';'.join([os.path.abspath(os.path.join(base_folder, path)) for path in path_l])

train_data[image_col] = train_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
test_data[image_col] = test_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
```

## Training

Train the MultiModalPredictor with a time limit:

```python
from autogluon.multimodal import MultiModalPredictor

predictor = MultiModalPredictor(label=label_col).fit(
    train_data=train_data,
    time_limit=120  # 2 minutes time limit for quick demo
)
```

Key features:
- Automatically infers problem type (classification/regression)
- Detects feature modalities
- Selects appropriate models
- Adds late-fusion model (MLP or transformer) for multiple backbones

## Prediction

Make predictions on test data:

```python
# Class predictions
predictions = predictor.predict(test_data.drop(columns=label_col))

# Probability predictions
probs = predictor.predict_proba(test_data.drop(columns=label_col))
```

## Evaluation

Evaluate model performance:

```python
scores = predictor.evaluate(test_data, metrics=["roc_auc"])
```

This tutorial demonstrates basic functionality - see in-depth tutorials for advanced features like embedding extraction, distillation, model fine-tuning, and semantic matching.