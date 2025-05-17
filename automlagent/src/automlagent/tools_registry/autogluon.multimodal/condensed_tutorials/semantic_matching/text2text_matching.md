# Condensed: Text-to-Text Semantic Matching with AutoMM

Summary: This tutorial demonstrates implementing text-to-text semantic matching using AutoGluon's MultiModalPredictor. It covers techniques for computing similarity between text pairs using BERT embeddings, specifically useful for tasks like web search, QA, and document deduplication. Key functionalities include data preparation with SNLI dataset, model training configuration with text similarity settings, making predictions on text pairs, and extracting embeddings. The tutorial shows how to set up binary classification for semantic matching, configure match labels, and evaluate model performance using AUC metrics, all through AutoMM's streamlined interface.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details:

# Text-to-Text Semantic Matching with AutoMM

## Key Concepts
- Computes similarity between text pairs using semantic matching
- Uses BERT to project sentences into vectors for classification
- Suitable for tasks like web search, QA, document deduplication, etc.

## Implementation

### 1. Data Preparation
```python
from autogluon.core.utils.loaders import load_pd

# Load SNLI dataset
snli_train = load_pd.load('https://automl-mm-bench.s3.amazonaws.com/snli/snli_train.csv', delimiter="|")
snli_test = load_pd.load('https://automl-mm-bench.s3.amazonaws.com/snli/snli_test.csv', delimiter="|")
```

### 2. Model Training
```python
from autogluon.multimodal import MultiModalPredictor

predictor = MultiModalPredictor(
    problem_type="text_similarity",
    query="premise",          # first sentence column
    response="hypothesis",    # second sentence column
    label="label",           # label column
    match_label=1,           # label indicating semantic match
    eval_metric='auc'
)

predictor.fit(
    train_data=snli_train,
    time_limit=180
)
```

### 3. Making Predictions

```python
# Single prediction
pred_data = pd.DataFrame({
    "premise": ["The teacher gave his speech to an empty room."], 
    "hypothesis": ["There was almost nobody when the professor was talking."]
})

# Get predictions
predictions = predictor.predict(pred_data)
probabilities = predictor.predict_proba(pred_data)

# Extract embeddings
embeddings_1 = predictor.extract_embedding({"premise": ["The teacher gave his speech to an empty room."]})
embeddings_2 = predictor.extract_embedding({"hypothesis": ["There was almost nobody when the professor was talking."]})
```

## Important Configurations
- `problem_type`: Set to "text_similarity"
- `match_label`: Specify which label represents semantic matching (typically 1)
- `eval_metric`: Uses 'auc' for evaluation

## Best Practices
1. Labels should be binary
2. Define `match_label` based on specific task context
3. Use appropriate column names for query and response
4. Consider task-specific requirements when setting evaluation metrics

## Evaluation
```python
score = predictor.evaluate(snli_test)
```

This implementation provides semantic matching capabilities using BERT-based embeddings through AutoMM's simplified interface.