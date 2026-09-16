# Condensed: ```python

Summary: This tutorial demonstrates implementing image-to-image semantic matching using AutoGluon's MultiModalPredictor with Swin Transformer architecture. It covers: (1) preparing paired image datasets with binary labels for similarity tasks, (2) expanding relative file paths to absolute paths for data loading, (3) training image similarity models by specifying query/response columns and match_label parameter, (4) evaluating models with ROC-AUC metrics, (5) generating binary predictions and probability scores with custom thresholding, and (6) extracting high-dimensional image embeddings for similarity computation. Key implementation details include configuring problem_type="image_similarity", setting match_label to identify positive pairs, and using cosine similarity for comparing image feature vectors. Useful for building image matching, duplicate detection, and visual similarity search systems.

*This is a condensed version that preserves essential implementation details and context.*

# Image-to-Image Semantic Matching with AutoMM

## Installation & Setup
```python
!pip install autogluon.multimodal
import os
import pandas as pd
from autogluon.multimodal import MultiModalPredictor
```

## Data Preparation

**Dataset**: Stanford Online Products (SOP) - 12 product categories with multiple views per product
- **Positive pairs (label=1)**: Different views of same product
- **Negative pairs (label=0)**: Images from different products

```python
# Download and load dataset
download_dir = './ag_automm_tutorial_img2img'
zip_file = 'https://automl-mm-bench.s3.amazonaws.com/Stanford_Online_Products.zip'
from autogluon.core.utils.loaders import load_zip
load_zip.unzip(zip_file, unzip_dir=download_dir)

dataset_path = os.path.join(download_dir, 'Stanford_Online_Products')
train_data = pd.read_csv(f'{dataset_path}/train.csv', index_col=0)
test_data = pd.read_csv(f'{dataset_path}/test.csv', index_col=0)

# Define column names
image_col_1 = "Image1"
image_col_2 = "Image2"
label_col = "Label"
match_label = 1  # Label indicating semantic match
```

**⚠️ Critical**: Specify `match_label` based on your task context (the label class representing matching pairs).

```python
# Expand relative image paths to absolute paths
def path_expander(path, base_folder):
    path_l = path.split(';')
    return ';'.join([os.path.abspath(os.path.join(base_folder, path)) for path in path_l])

for image_col in [image_col_1, image_col_2]:
    train_data[image_col] = train_data[image_col].apply(
        lambda ele: path_expander(ele, base_folder=dataset_path))
    test_data[image_col] = test_data[image_col].apply(
        lambda ele: path_expander(ele, base_folder=dataset_path))
```

## Model Training

AutoMM uses **Swin Transformer** to project images into high-dimensional vectors and computes cosine similarity.

```python
predictor = MultiModalPredictor(
    problem_type="image_similarity",
    query=image_col_1,           # First image column
    response=image_col_2,         # Second image column
    label=label_col,              # Label column
    match_label=match_label,      # Label for matching pairs
    eval_metric='auc',            # Evaluation metric
)

predictor.fit(train_data=train_data, time_limit=180)
```

## Evaluation & Prediction

```python
# Evaluate with ROC-AUC
score = predictor.evaluate(test_data)

# Binary predictions (threshold=0.5)
pred = predictor.predict(test_data.head(3))

# Get probabilities for custom thresholding
proba = predictor.predict_proba(test_data.head(3))
```

## Extract Embeddings

```python
# Extract embeddings for each image separately
embeddings_1 = predictor.extract_embedding({image_col_1: test_data[image_col_1][:5].tolist()})
embeddings_2 = predictor.extract_embedding({image_col_2: test_data[image_col_2][:5].tolist()})
print(embeddings_1.shape)  # Shape: (num_images, embedding_dim)
```

## Additional Resources
- [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm)
- [Customize AutoMM](../advanced_topics/customization.ipynb)