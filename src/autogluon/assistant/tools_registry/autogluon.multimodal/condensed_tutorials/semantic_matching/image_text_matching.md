# Condensed: ```python

Summary: This tutorial demonstrates implementing image-text matching using AutoGluon's MultiModalPredictor with CLIP backbone. It covers: (1) dataset preparation with pandas dataframes for image-text pairs, (2) initializing predictors with `problem_type="image_text_similarity"`, (3) zero-shot and finetuned model evaluation using recall metrics, (4) predicting match probabilities between images and text, (5) extracting embeddings for both modalities, and (6) bidirectional semantic search (text-to-image and image-to-text retrieval). Key techniques include path expansion for image files, preparing evaluation data with unique items, configuring query/response columns, and using `semantic_search()` utility for retrieval tasks. Useful for building cross-modal retrieval systems, similarity matching applications, and multimodal embedding extraction.

*This is a condensed version that preserves essential implementation details and context.*

# Image-Text Matching with AutoGluon MultiModal

## Installation & Setup

```python
!pip install autogluon.multimodal

import os
import warnings
from IPython.display import Image, display
import numpy as np
warnings.filterwarnings('ignore')
np.random.seed(123)
```

## Dataset Preparation

**Dataset**: Flickr30K - 31,783 images with descriptive captions, organized in pandas dataframes.

```python
from autogluon.core.utils.loaders import load_zip
import pandas as pd

# Download dataset
download_dir = './ag_automm_tutorial_imgtxt'
zip_file = 'https://automl-mm-bench.s3.amazonaws.com/flickr30k.zip'
load_zip.unzip(zip_file, unzip_dir=download_dir)

# Load data
dataset_path = os.path.join(download_dir, 'flickr30k_processed')
train_data = pd.read_csv(f'{dataset_path}/train.csv', index_col=0)
val_data = pd.read_csv(f'{dataset_path}/val.csv', index_col=0)
test_data = pd.read_csv(f'{dataset_path}/test.csv', index_col=0)
image_col = "image"
text_col = "caption"

# Expand relative paths to absolute paths
def path_expander(path, base_folder):
    path_l = path.split(';')
    return ';'.join([os.path.abspath(os.path.join(base_folder, path)) for path in path_l])

train_data[image_col] = train_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
val_data[image_col] = val_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
test_data[image_col] = test_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
```

**Data Structure**: Each row contains one image-text pair. Since each image has 5 captions, image paths are duplicated 5 times.

**Prepare evaluation data**:
```python
test_image_data = pd.DataFrame({image_col: test_data[image_col].unique().tolist()})
test_text_data = pd.DataFrame({text_col: test_data[text_col].unique().tolist()})
test_data_with_label = test_data.copy()
test_label_col = "relevance"
test_data_with_label[test_label_col] = [1] * len(test_data)
```

## Initialize Predictor

**Critical Configuration**:
- `problem_type="image_text_similarity"` - Required for image-text matching
- `query` and `response` - Specify dataframe columns (interchangeable for image-text matching)
- Automatically loads pretrained CLIP backbone: `openai/clip-vit-base-patch32`

```python
from autogluon.multimodal import MultiModalPredictor

predictor = MultiModalPredictor(
    query=text_col,
    response=image_col,
    problem_type="image_text_similarity",
    eval_metric="recall",
)
```

## Zero-Shot Evaluation

Evaluate pretrained model without finetuning:

```python
txt_to_img_scores = predictor.evaluate(
    data=test_data_with_label,
    query_data=test_text_data,
    response_data=test_image_data,
    label=test_label_col,
    cutoffs=[1, 5, 10],
)
img_to_txt_scores = predictor.evaluate(
    data=test_data_with_label,
    query_data=test_image_data,
    response_data=test_text_data,
    label=test_label_col,
    cutoffs=[1, 5, 10],
)
```

**Note**: Text-to-image recalls are typically higher than image-to-text due to 5 captions per image. Image-to-text `recall@1` upper bound is 20%.

## Finetuning

```python
predictor.fit(
    train_data=train_data,
    tuning_data=val_data,
    time_limit=180,  # seconds
)
```

Evaluate finetuned model using same `.evaluate()` calls as above. Expect significant improvements over zero-shot.

## Key Prediction Methods

**Match Prediction**:
```python
pred = predictor.predict(test_data.head(5))
```

**Matching Probabilities** (2nd column = match probability):
```python
proba = predictor.predict_proba(test_data.head(5))
```

**Extract Embeddings**:
```python
image_embeddings = predictor.extract_embedding({image_col: test_image_data[image_col][:5].tolist()})
text_embeddings = predictor.extract_embedding({text_col: test_text_data[text_col][:5].tolist()})
```

## Semantic Search

**Text-to-Image Search**:
```python
from autogluon.multimodal.utils import semantic_search

text_to_image_hits = semantic_search(
    matcher=predictor,
    query_data=test_text_data.iloc[[3]],
    response_data=test_image_data,
    top_k=5,
)
```

**Image-to-Text Search**:
```python
image_to_text_hits = semantic_search(
    matcher=predictor,
    query_data=test_image_data.iloc[[6]],
    response_data=test_text_data,
    top_k=5,
)
```

## Additional Resources

- **More Examples**: [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm)
- **Customization**: [Customize AutoMM](../advanced_topics/customization.ipynb)