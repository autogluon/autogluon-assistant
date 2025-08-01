# Condensed: ```python

Summary: This tutorial demonstrates implementing text similarity models with AutoGluon MultiModal, teaching how to leverage BERT-based sentence embeddings for semantic matching tasks. It covers loading text pair datasets, configuring and training a text similarity predictor with customizable parameters (query/response columns, match labels), evaluating model performance, making predictions on new sentence pairs, and extracting embeddings from individual sentences. Key functionalities include binary classification for semantic similarity, probability score generation, and embedding extraction—valuable for applications like duplicate detection, paraphrase identification, and semantic search implementations.

*This is a condensed version that preserves essential implementation details and context.*

# Text Similarity with AutoGluon MultiModal

## Setup and Data Loading

```python
!pip install autogluon.multimodal

from autogluon.core.utils.loaders import load_pd
import pandas as pd

snli_train = load_pd.load('https://automl-mm-bench.s3.amazonaws.com/snli/snli_train.csv', delimiter="|")
snli_test = load_pd.load('https://automl-mm-bench.s3.amazonaws.com/snli/snli_test.csv', delimiter="|")
```

## Training the Model

AutoGluon MultiModal uses BERT to project sentences into high-dimensional vectors and treats matching as a classification problem following the sentence transformers approach.

```python
from autogluon.multimodal import MultiModalPredictor

predictor = MultiModalPredictor(
    problem_type="text_similarity",
    query="premise",           # first sentence column
    response="hypothesis",     # second sentence column
    label="label",             # label column
    match_label=1,             # label indicating semantic match
    eval_metric='auc',
)

predictor.fit(
    train_data=snli_train,
    time_limit=180,
)
```

## Evaluation and Prediction

```python
# Evaluate on test data
score = predictor.evaluate(snli_test)
print("evaluation score: ", score)

# Predict on new sentence pair
pred_data = pd.DataFrame.from_dict({
    "premise": ["The teacher gave his speech to an empty room."], 
    "hypothesis": ["There was almost nobody when the professor was talking."]
})

# Get predictions
predictions = predictor.predict(pred_data)
print('Predicted entities:', predictions[0])

# Get matching probabilities
probabilities = predictor.predict_proba(pred_data)
print(probabilities)
```

## Extracting Embeddings

```python
# Extract embeddings for individual sentences
embeddings_1 = predictor.extract_embedding({"premise": ["The teacher gave his speech to an empty room."]})
print(embeddings_1.shape)

embeddings_2 = predictor.extract_embedding({"hypothesis": ["There was almost nobody when the professor was talking."]})
print(embeddings_2.shape)
```

**Key Implementation Notes:**
- Labels must be binary with `match_label` specifying which value indicates semantic similarity
- The model uses BERT to create sentence embeddings for semantic matching
- For custom tasks, define `match_label` based on your specific context (e.g., duplicate/not duplicate)
- For customization options, refer to "Customize AutoMM" documentation