# Condensed: ```python

Summary: This tutorial demonstrates AutoGluon MultiModal, a framework for multimodal machine learning that handles mixed data types. It covers implementing a pet adoption speed predictor using images and tabular data, showcasing automatic modality detection, model selection, and late-fusion techniques. Key functionalities include training a multimodal predictor with minimal code, evaluating model performance, making predictions, extracting embeddings for downstream tasks, and saving/loading models. The tutorial is valuable for developers implementing classification tasks with mixed data types (images and tabular data) who need automated machine learning capabilities without extensive manual configuration.

*This is a condensed version that preserves essential implementation details and context.*

# AutoGluon MultiModal Tutorial

## Setup

```python
!pip install autogluon.multimodal

import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')
np.random.seed(123)
```

## Dataset Preparation

Using a simplified version of PetFinder dataset to predict animal adoption rates (0=slow, 1=fast).

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

# Process image paths
image_col = 'Images'
train_data[image_col] = train_data[image_col].apply(lambda ele: ele.split(';')[0])  # Use first image only
test_data[image_col] = test_data[image_col].apply(lambda ele: ele.split(';')[0])

def path_expander(path, base_folder):
    path_l = path.split(';')
    return ';'.join([os.path.abspath(os.path.join(base_folder, path)) for path in path_l])

train_data[image_col] = train_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
test_data[image_col] = test_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
```

## Training

```python
from autogluon.multimodal import MultiModalPredictor
predictor = MultiModalPredictor(label=label_col)
predictor.fit(
    train_data=train_data,
    time_limit=120,  # seconds
)
```

AutoMM automatically:
- Infers problem type (classification/regression)
- Detects data modalities
- Selects appropriate models
- Trains with late-fusion if multiple backbones are available

## Evaluation

```python
scores = predictor.evaluate(test_data, metrics=["roc_auc"])
```

## Prediction

```python
# Get class predictions
predictions = predictor.predict(test_data.drop(columns=label_col))

# Get class probabilities (classification only)
probas = predictor.predict_proba(test_data.drop(columns=label_col))
```

## Extract Embeddings

```python
embeddings = predictor.extract_embedding(test_data.drop(columns=label_col))
```

## Save and Load

```python
import uuid
model_path = f"./tmp/{uuid.uuid4().hex}-saved_model"
predictor.save(model_path)
loaded_predictor = MultiModalPredictor.load(model_path)
```

⚠️ **Warning**: `MultiModalPredictor.load()` uses `pickle` module which can be insecure. Only load models from trusted sources.

## Additional Resources
- More examples: [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm)
- Customization: Refer to "Customize AutoMM" documentation