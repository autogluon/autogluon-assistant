# Condensed: ```python

Summary: This tutorial demonstrates image-to-image semantic matching using AutoGluon's MultiModalPredictor. It covers implementing similarity-based image matching with Swin Transformer embeddings and cosine similarity calculations. Key functionalities include: data preparation with image path handling, model training with the "image_similarity" problem type, evaluation using AUC metrics, prediction with probability outputs for custom thresholding, and feature extraction to obtain image embeddings. The tutorial helps with tasks like determining if two product images represent the same item, building image retrieval systems, and creating custom similarity thresholds for matching applications.

*This is a condensed version that preserves essential implementation details and context.*

# Image-to-Image Semantic Matching with AutoMM

## Setup

```python
!pip install autogluon.multimodal

import os
import pandas as pd
import warnings
from IPython.display import Image, display
warnings.filterwarnings('ignore')
```

## Data Preparation

This tutorial uses the simplified Stanford Online Products dataset (SOP) for image-to-image semantic matching:

```python
# Download dataset
download_dir = './ag_automm_tutorial_img2img'
zip_file = 'https://automl-mm-bench.s3.amazonaws.com/Stanford_Online_Products.zip'
from autogluon.core.utils.loaders import load_zip
load_zip.unzip(zip_file, unzip_dir=download_dir)

# Load annotations
dataset_path = os.path.join(download_dir, 'Stanford_Online_Products')
train_data = pd.read_csv(f'{dataset_path}/train.csv', index_col=0)
test_data = pd.read_csv(f'{dataset_path}/test.csv', index_col=0)
image_col_1 = "Image1"
image_col_2 = "Image2"
label_col = "Label"
match_label = 1  # Label class representing a semantic match

# Expand image paths
def path_expander(path, base_folder):
    path_l = path.split(';')
    return ';'.join([os.path.abspath(os.path.join(base_folder, path)) for path in path_l])

for image_col in [image_col_1, image_col_2]:
    train_data[image_col] = train_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
    test_data[image_col] = test_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
```

## Model Training

AutoMM uses Swin Transformer to project images into high-dimensional vectors and computes cosine similarity:

```python
from autogluon.multimodal import MultiModalPredictor
predictor = MultiModalPredictor(
    problem_type="image_similarity",
    query=image_col_1,         # first image column
    response=image_col_2,      # second image column
    label=label_col,           # label column
    match_label=match_label,   # label indicating semantic match
    eval_metric='auc',         # evaluation metric
)
    
# Fit the model
predictor.fit(
    train_data=train_data,
    time_limit=180,
)
```

## Evaluation and Prediction

```python
# Evaluate on test data
score = predictor.evaluate(test_data)
print("evaluation score: ", score)

# Predict on image pairs (using 0.5 threshold)
pred = predictor.predict(test_data.head(3))
print(pred)

# Get probabilities for custom thresholding
proba = predictor.predict_proba(test_data.head(3))
print(proba)
```

## Feature Extraction

Extract embeddings for each image:

```python
embeddings_1 = predictor.extract_embedding({image_col_1: test_data[image_col_1][:5].tolist()})
print(embeddings_1.shape)
embeddings_2 = predictor.extract_embedding({image_col_2: test_data[image_col_2][:5].tolist()})
print(embeddings_2.shape)
```

For more examples, see [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm).
For customization options, refer to [Customize AutoMM](../advanced_topics/customization.ipynb).