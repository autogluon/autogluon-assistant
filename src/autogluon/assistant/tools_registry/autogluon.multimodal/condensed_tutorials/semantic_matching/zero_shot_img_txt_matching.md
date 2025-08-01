# Condensed: ```python

Summary: This tutorial demonstrates using AutoGluon MultiModal for image-text similarity tasks. It covers extracting embeddings from images and text, performing cross-modal retrieval (finding images matching text queries and vice versa using semantic search), and predicting image-text matching with probability scores. Key functionalities include embedding extraction, semantic search for retrieval, and binary classification of image-text pairs. The implementation uses MultiModalPredictor with "image_text_similarity" problem type, making it valuable for building multimodal search systems, content recommendation, or image captioning validation.

*This is a condensed version that preserves essential implementation details and context.*

# Image-Text Similarity with AutoGluon MultiModal

## Setup and Data Preparation

```python
!pip install autogluon.multimodal

from autogluon.multimodal.utils import download

# Sample texts
texts = [
    "A cheetah chases prey on across a field.",
    "A man is eating a piece of bread.",
    # ... more text examples ...
    "A monkey is playing drums.",
]

# Sample image URLs
urls = ['http://farm4.staticflickr.com/3179/2872917634_f41e6987a8_z.jpg',
        # ... more URLs ...
        'https://farm6.staticflickr.com/5251/5548123650_1a69ce1e34_z.jpg']

image_paths = [download(url) for url in urls]
```

## Extract Embeddings

Initialize the predictor with the appropriate problem type:

```python
from autogluon.multimodal import MultiModalPredictor
predictor = MultiModalPredictor(problem_type="image_text_similarity")

# Extract embeddings
image_embeddings = predictor.extract_embedding(image_paths, as_tensor=True)
text_embeddings = predictor.extract_embedding(texts, as_tensor=True)
```

## Image Retrieval with Text Query

Retrieve images that match a text query:

```python
from autogluon.multimodal.utils import semantic_search

# Search for images matching a text query
hits = semantic_search(
    matcher=predictor,
    query_embeddings=text_embeddings[6][None,],  # "There is a carriage in the image."
    response_embeddings=image_embeddings,
    top_k=5,
)
```

## Text Retrieval with Image Query

Retrieve texts that match an image query:

```python
# Search for texts matching an image query
hits = semantic_search(
    matcher=predictor,
    query_embeddings=image_embeddings[4][None,],  # Image of a man riding a horse
    response_embeddings=text_embeddings,
    top_k=5,
)
```

## Predict Image-Text Matching

Initialize a predictor for matching prediction:

```python
predictor = MultiModalPredictor(
    query="abc",
    response="xyz",
    problem_type="image_text_similarity",
)

# Predict if an image-text pair matches
pred = predictor.predict({"abc": [image_paths[4]], "xyz": [texts[3]]})

# Get matching probabilities
proba = predictor.predict_proba({"abc": [image_paths[4]], "xyz": [texts[3]]})
```

## Key Points

- Use `problem_type="image_text_similarity"` for image-text similarity tasks
- Extract embeddings with `predictor.extract_embedding()`
- Use `semantic_search()` for retrieval tasks
- For direct matching prediction, initialize with `query` and `response` parameters
- Both binary predictions and probability scores are available

For customization options, refer to the "Customize AutoMM" documentation.