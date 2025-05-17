# Condensed: Image-to-Image Semantic Matching with AutoMM

Summary: This tutorial demonstrates implementing image-to-image semantic matching using AutoGluon's MultiModalPredictor. It covers essential techniques for training a model that determines if two images are semantically similar, using the Stanford Online Products dataset. Key implementations include data preparation with path handling, model configuration for image similarity tasks, training with customizable parameters, and various prediction methods (binary classification, probability scores, and embedding extraction). The tutorial showcases how to leverage Swin Transformer for feature vector generation and cosine similarity computation, making it valuable for tasks involving image pair matching, product similarity detection, and feature extraction for downstream applications.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details:

# Image-to-Image Semantic Matching with AutoMM

## Key Implementation Details

### Setup
```python
!pip install autogluon.multimodal
import pandas as pd
from autogluon.multimodal import MultiModalPredictor
```

### Data Preparation
1. Dataset structure:
   - Two image columns (Image1, Image2)
   - Label column (1 = matching pair, 0 = non-matching pair)
   - Uses Stanford Online Products dataset with 12 product categories

2. Path handling:
```python
def path_expander(path, base_folder):
    path_l = path.split(';')
    return ';'.join([os.path.abspath(os.path.join(base_folder, path)) for path in path_l])

# Apply to both image columns
for image_col in [image_col_1, image_col_2]:
    train_data[image_col] = train_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
```

### Model Training
```python
predictor = MultiModalPredictor(
    problem_type="image_similarity",
    query=image_col_1,          # first image column
    response=image_col_2,       # second image column
    label=label_col,           # label column
    match_label=match_label,   # label value indicating matching pairs (e.g., 1)
    eval_metric='auc'          # evaluation metric
)

predictor.fit(
    train_data=train_data,
    time_limit=180
)
```

### Model Usage

1. Evaluation:
```python
score = predictor.evaluate(test_data)
```

2. Prediction:
```python
# Binary predictions (threshold = 0.5)
predictions = predictor.predict(test_data)

# Probability scores
probabilities = predictor.predict_proba(test_data)
```

3. Feature Extraction:
```python
# Extract embeddings for images
embeddings_1 = predictor.extract_embedding({image_col_1: test_data[image_col_1][:5].tolist()})
embeddings_2 = predictor.extract_embedding({image_col_2: test_data[image_col_2][:5].tolist()})
```

## Important Notes
- Model uses Swin Transformer to project images into feature vectors
- Similarity is computed using cosine similarity between feature vectors
- Default prediction threshold is 0.5 for binary classification
- Embeddings can be extracted separately for each image

## Best Practices
1. Ensure image paths are properly formatted and accessible
2. Consider task context when specifying `match_label`
3. Adjust time_limit based on dataset size and computational resources
4. Use predict_proba() for custom thresholding if needed