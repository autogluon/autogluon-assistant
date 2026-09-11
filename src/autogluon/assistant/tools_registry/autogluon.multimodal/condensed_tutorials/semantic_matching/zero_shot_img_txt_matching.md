# Condensed: ```python

Summary: This tutorial covers AutoGluon MultiModal's image-text similarity capabilities for building multimodal search and matching systems. It demonstrates extracting embeddings from images and text using `MultiModalPredictor` with `problem_type="image_text_similarity"`, implementing bidirectional semantic search (text-to-image and image-to-text retrieval) using `semantic_search()` with cosine similarity, and predicting image-text pair matches with binary predictions and probabilities. Key techniques include offline embedding extraction, configuring query/response parameters for pair matching, using `as_tensor=True` for tensor outputs, and leveraging `top_k` for ranked retrieval results. Useful for tasks like image database search, content recommendation, visual question answering preprocessing, and multimodal classification systems.

*This is a condensed version that preserves essential implementation details and context.*

# Image-Text Similarity with AutoGluon MultiModal

## Installation & Setup

```python
!pip install autogluon.multimodal
from autogluon.multimodal import MultiModalPredictor
```

## Extract Embeddings

Initialize predictor with `image_text_similarity` problem type:

```python
predictor = MultiModalPredictor(problem_type="image_text_similarity")
```

Extract embeddings separately for images and text:

```python
# Extract image embeddings
image_embeddings = predictor.extract_embedding(image_paths, as_tensor=True)

# Extract text embeddings  
text_embeddings = predictor.extract_embedding(texts, as_tensor=True)
```

## Image Retrieval with Text Query

Use `semantic_search` to find images matching text queries:

```python
from autogluon.multimodal.utils import semantic_search

hits = semantic_search(
    matcher=predictor,
    query_embeddings=text_embeddings[6][None,],
    response_embeddings=image_embeddings,
    top_k=5,
)
```

**Workflow**: Extract image embeddings offline → Extract text query embedding → Compute cosine similarities → Return top candidates

## Text Retrieval with Image Query

Switch `query_embeddings` and `response_embeddings` assignments:

```python
hits = semantic_search(
    matcher=predictor,
    query_embeddings=image_embeddings[4][None,],
    response_embeddings=text_embeddings,
    top_k=5,
)
```

## Predict Image-Text Pair Matching

Initialize predictor with `query` and `response` parameters:

```python
predictor = MultiModalPredictor(
    query="abc",
    response="xyz",
    problem_type="image_text_similarity",
)

# Binary prediction
pred = predictor.predict({"abc": [image_paths[4]], "xyz": [texts[3]]})

# Probability prediction
proba = predictor.predict_proba({"abc": [image_paths[4]], "xyz": [texts[3]]})
```

**Note**: Use probabilities with custom thresholds for flexible matching decisions.

## Additional Resources

- [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm)
- [Customize AutoMM](../advanced_topics/customization.ipynb)