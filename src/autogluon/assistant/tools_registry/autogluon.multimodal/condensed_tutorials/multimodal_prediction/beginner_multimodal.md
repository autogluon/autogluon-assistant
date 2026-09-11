# Condensed: ```python

Summary: This tutorial demonstrates AutoGluon's MultiModalPredictor for building classification/regression models using combined image, text, and tabular data. Key implementations include: preparing multimodal datasets with proper image path expansion to absolute paths, automatic model training with time limits, evaluation with custom metrics (e.g., roc_auc), generating predictions and class probabilities, extracting embeddings for feature representation, and model persistence via save/load (with pickle security warnings). The tutorial helps with tasks like multimodal data preprocessing, automated ML pipeline setup with minimal code, handling mixed data types in pandas DataFrames, and deploying multimodal predictors. Core features covered: automatic problem type inference, modality detection, late-fusion architecture, and seamless integration of images/text/tabular features without manual feature engineering.

*This is a condensed version that preserves essential implementation details and context.*

# AutoGluon MultiModal Tutorial (Condensed)

## Installation & Setup
```python
!pip install autogluon.multimodal

import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')
np.random.seed(123)
```

## Dataset Preparation

**Task**: Predict animal adoption rates (binary: 0=slow, 1=fast) using multimodal data (images, text, tabular features) from PetFinder dataset.

```python
# Download dataset
download_dir = './ag_automm_tutorial'
zip_file = 'https://automl-mm-bench.s3.amazonaws.com/petfinder_for_tutorial.zip'
from autogluon.core.utils.loaders import load_zip
load_zip.unzip(zip_file, unzip_dir=download_dir)

# Load data
import pandas as pd
dataset_path = download_dir + '/petfinder_for_tutorial'
train_data = pd.read_csv(f'{dataset_path}/train.csv', index_col=0)
test_data = pd.read_csv(f'{dataset_path}/test.csv', index_col=0)
label_col = 'AdoptionSpeed'
```

**Critical**: Expand image paths to absolute paths:
```python
image_col = 'Images'
# Use first image only
train_data[image_col] = train_data[image_col].apply(lambda ele: ele.split(';')[0])
test_data[image_col] = test_data[image_col].apply(lambda ele: ele.split(';')[0])

# Expand to absolute paths
def path_expander(path, base_folder):
    path_l = path.split(';')
    return ';'.join([os.path.abspath(os.path.join(base_folder, path)) for path in path_l])

train_data[image_col] = train_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
test_data[image_col] = test_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
```

## Training

**Key Feature**: AutoMM automatically handles problem type inference, modality detection, model selection, and late-fusion (MLP/transformer) when multiple backbones exist.

```python
from autogluon.multimodal import MultiModalPredictor
predictor = MultiModalPredictor(label=label_col)
predictor.fit(
    train_data=train_data,
    time_limit=120,  # seconds
)
```

## Evaluation & Prediction

```python
# Evaluate
scores = predictor.evaluate(test_data, metrics=["roc_auc"])

# Predict labels
predictions = predictor.predict(test_data.drop(columns=label_col))

# Get class probabilities (classification only)
probas = predictor.predict_proba(test_data.drop(columns=label_col))
```

**Note**: `.predict_proba()` throws exception on regression tasks.

## Extract Embeddings

```python
embeddings = predictor.extract_embedding(test_data.drop(columns=label_col))
```

## Save & Load

```python
model_path = f"./tmp/{uuid.uuid4().hex}-saved_model"
predictor.save(model_path)
loaded_predictor = MultiModalPredictor.load(model_path)
```

**⚠️ Security Warning**: `MultiModalPredictor.load()` uses `pickle` internally. **Only load data from trusted sources** - malicious pickle data can execute arbitrary code during unpickling.

## Additional Resources
- [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm)
- [Customize AutoMM](../advanced_topics/customization.ipynb)