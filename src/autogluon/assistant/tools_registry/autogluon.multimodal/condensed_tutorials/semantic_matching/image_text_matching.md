# Condensed: ```python

Summary: This tutorial demonstrates implementing image-text matching with AutoGluon MultiModal, covering techniques for bidirectional retrieval between images and text. It shows how to prepare datasets, initialize a MultiModalPredictor for image-text similarity tasks, perform zero-shot evaluation, finetune models, and evaluate performance using recall metrics. Key functionalities include extracting embeddings from both modalities, making match predictions with confidence scores, and performing semantic search (text-to-image and image-to-text). The tutorial provides a complete workflow for building cross-modal retrieval systems that can find relevant images given text queries and vice versa.

*This is a condensed version that preserves essential implementation details and context.*

# Image-Text Matching with AutoGluon MultiModal

## Setup

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

```python
# Download and extract Flickr30K dataset
from autogluon.core.utils.loaders import load_pd, load_zip
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

# Convert relative paths to absolute paths
def path_expander(path, base_folder):
    path_l = path.split(';')
    return ';'.join([os.path.abspath(os.path.join(base_folder, path)) for path in path_l])

train_data[image_col] = train_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
val_data[image_col] = val_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
test_data[image_col] = test_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))

# Prepare evaluation data
test_image_data = pd.DataFrame({image_col: test_data[image_col].unique().tolist()})
test_text_data = pd.DataFrame({text_col: test_data[text_col].unique().tolist()})
test_data_with_label = test_data.copy()
test_label_col = "relevance"
test_data_with_label[test_label_col] = [1] * len(test_data)
```

## Initialize Predictor

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

```python
# Text-to-image retrieval
txt_to_img_scores = predictor.evaluate(
    data=test_data_with_label,
    query_data=test_text_data,
    response_data=test_image_data,
    label=test_label_col,
    cutoffs=[1, 5, 10],
)

# Image-to-text retrieval
img_to_txt_scores = predictor.evaluate(
    data=test_data_with_label,
    query_data=test_image_data,
    response_data=test_text_data,
    label=test_label_col,
    cutoffs=[1, 5, 10],
)
print(f"txt_to_img_scores: {txt_to_img_scores}")
print(f"img_to_txt_scores: {img_to_txt_scores}")
```

## Finetune the Model

```python
predictor.fit(
    train_data=train_data,
    tuning_data=val_data,
    time_limit=180,  # Quick demo with 3 minutes
)
```

## Evaluate Finetuned Model

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
print(f"txt_to_img_scores: {txt_to_img_scores}")
print(f"img_to_txt_scores: {img_to_txt_scores}")
```

## Prediction Functions

```python
# Predict match/no-match
pred = predictor.predict(test_data.head(5))
print(pred)

# Predict matching probabilities
proba = predictor.predict_proba(test_data.head(5))
print(proba)  # Second column is the probability of being a match
```

## Extract Embeddings

```python
# Extract image embeddings
image_embeddings = predictor.extract_embedding({image_col: test_image_data[image_col][:5].tolist()})
print(image_embeddings.shape)

# Extract text embeddings
text_embeddings = predictor.extract_embedding({text_col: test_text_data[text_col][:5].tolist()})
print(text_embeddings.shape)
```

## Semantic Search

```python
from autogluon.multimodal.utils import semantic_search

# Text-to-image search
text_to_image_hits = semantic_search(
    matcher=predictor,
    query_data=test_text_data.iloc[[3]],
    response_data=test_image_data,
    top_k=5,
)

# Image-to-text search
image_to_text_hits = semantic_search(
    matcher=predictor,
    query_data=test_image_data.iloc[[6]],
    response_data=test_text_data,
    top_k=5,
)
```

For more examples, see [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm).
For customization options, refer to [Customize AutoMM](../advanced_topics/customization.ipynb).