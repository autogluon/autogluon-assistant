# Condensed: Image-Text Semantic Matching with AutoMM - Zero-Shot

Summary: This tutorial demonstrates implementing zero-shot image-text semantic matching using AutoMM's MultiModalPredictor with CLIP model. It covers three main functionalities: image retrieval using text queries, text retrieval using image queries, and direct image-text pair matching. Key implementation techniques include efficient embedding extraction with tensor output, offline embedding storage for scalability, and semantic search functionality. The tutorial helps with tasks like building image search systems, text-based image retrieval, and determining semantic similarity between image-text pairs. Notable features include batch processing support, two-tower architecture utilization, and flexible matching threshold configuration through probability scores.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details:

# Zero-Shot Image-Text Semantic Matching with AutoMM

## Key Concepts
- Uses CLIP model for image-text matching without training
- Two-tower architecture: separate encoders for images and text
- Enables offline embedding extraction for scalability
- Supports image retrieval, text retrieval, and pair matching

## Implementation

### 1. Basic Setup
```python
from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.utils import semantic_search

# Initialize predictor
predictor = MultiModalPredictor(problem_type="image_text_similarity")
```

### 2. Embedding Extraction
```python
# Extract embeddings
image_embeddings = predictor.extract_embedding(image_paths, as_tensor=True)
text_embeddings = predictor.extract_embedding(texts, as_tensor=True)
```

### 3. Image-Text Search

#### Image Retrieval with Text Query
```python
hits = semantic_search(
    matcher=predictor,
    query_embeddings=text_embeddings[query_idx][None,],
    response_embeddings=image_embeddings,
    top_k=5
)
```

#### Text Retrieval with Image Query
```python
hits = semantic_search(
    matcher=predictor,
    query_embeddings=image_embeddings[image_idx][None,],
    response_embeddings=text_embeddings,
    top_k=5
)
```

### 4. Pair Matching

```python
# Initialize predictor for pair matching
predictor = MultiModalPredictor(
    query="abc",
    response="xyz",
    problem_type="image_text_similarity"
)

# Predict match
pred = predictor.predict({
    "abc": [image_paths[4]], 
    "xyz": [texts[3]]
})

# Get matching probabilities
proba = predictor.predict_proba({
    "abc": [image_paths[4]], 
    "xyz": [texts[3]]
})
```

## Important Notes
- Embeddings can be extracted offline for better scalability
- Use `as_tensor=True` for efficient embedding extraction
- Specify `query` and `response` names for pair matching
- Can use custom thresholds on probabilities for matching decisions

## Best Practices
1. Extract embeddings once and reuse for multiple queries
2. Use batch processing for large-scale applications
3. Consider memory requirements when working with large datasets
4. Store embeddings efficiently for production use