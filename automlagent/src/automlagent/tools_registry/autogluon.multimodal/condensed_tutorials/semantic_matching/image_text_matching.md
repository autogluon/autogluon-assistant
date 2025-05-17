# Condensed: Image-Text Semantic Matching with AutoMM

Summary: This tutorial demonstrates implementing image-text semantic matching using AutoGluon's MultiModalPredictor. It covers essential techniques for processing image-text pairs, including dataset preparation with path expansion, model configuration for similarity tasks, and both zero-shot and fine-tuned prediction approaches. The tutorial enables tasks like semantic search, embedding extraction, and match prediction, featuring key functionalities such as text-to-image and image-to-text search with customizable top-k results. It specifically implements CLIP-based matching with evaluation using recall metrics, making it valuable for building multimodal search and matching systems.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details:

# Image-Text Semantic Matching with AutoMM

## Key Implementation Details

### Setup
```python
!pip install autogluon.multimodal
import pandas as pd
from autogluon.multimodal import MultiModalPredictor
```

### Dataset Preparation
1. Load Flickr30K dataset containing image-text pairs
2. Key columns:
   - `image`: Path to image file
   - `caption`: Text description
   - `relevance`: Label column (1 for matching pairs)

```python
# Expand image paths to absolute paths
def path_expander(path, base_folder):
    path_l = path.split(';')
    return ';'.join([os.path.abspath(os.path.join(base_folder, path)) for path in path_l])

# Apply to dataframes
train_data[image_col] = train_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
```

### Predictor Configuration
```python
predictor = MultiModalPredictor(
    query=text_col,
    response=image_col,
    problem_type="image_text_similarity",
    eval_metric="recall"
)
```

### Key Features

1. **Zero-shot Evaluation**
```python
txt_to_img_scores = predictor.evaluate(
    data=test_data_with_label,
    query_data=test_text_data,
    response_data=test_image_data,
    label=test_label_col,
    cutoffs=[1, 5, 10]
)
```

2. **Model Finetuning**
```python
predictor.fit(
    train_data=train_data,
    tuning_data=val_data,
    time_limit=180  # in seconds
)
```

3. **Prediction Methods**
```python
# Match prediction
pred = predictor.predict(test_data)

# Matching probabilities
proba = predictor.predict_proba(test_data)

# Extract embeddings
image_embeddings = predictor.extract_embedding({image_col: image_paths})
text_embeddings = predictor.extract_embedding({text_col: text_list})
```

4. **Semantic Search**
```python
from autogluon.multimodal.utils import semantic_search

# Text-to-image search
text_to_image_hits = semantic_search(
    matcher=predictor,
    query_data=test_text_data,
    response_data=test_image_data,
    top_k=5
)

# Image-to-text search
image_to_text_hits = semantic_search(
    matcher=predictor,
    query_data=test_image_data,
    response_data=test_text_data,
    top_k=5
)
```

## Important Notes
- Uses CLIP backbone (`openai/clip-vit-base-patch32`) by default
- Each image typically corresponds to multiple captions
- Recall@k is used as evaluation metric
- Text-to-image recalls may be higher than image-to-text recalls due to multiple captions per image
- Finetuning can significantly improve performance over zero-shot prediction

For customization options, refer to the AutoMM customization documentation.