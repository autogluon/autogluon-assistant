# Condensed: ```python

Summary: This tutorial demonstrates semantic text similarity implementation using AutoGluon's MultiModalPredictor with BERT-based sentence transformers. It covers training binary classification models to match sentence pairs by configuring query/response columns, match labels, and evaluation metrics. Key functionalities include: fitting models with time limits, evaluating with ROC-AUC scores, predicting binary matches and probabilities for sentence pairs, and extracting separate embeddings for individual sentences. Essential for tasks involving duplicate detection, semantic matching, paraphrase identification, and sentence similarity scoring. Critical configuration requires binary labels and defining match_label for semantic equivalence context.

*This is a condensed version that preserves essential implementation details and context.*

# Text Similarity with AutoMM

## Installation & Data Loading
```python
!pip install autogluon.multimodal

from autogluon.core.utils.loaders import load_pd
snli_train = load_pd.load('https://automl-mm-bench.s3.amazonaws.com/snli/snli_train.csv', delimiter="|")
snli_test = load_pd.load('https://automl-mm-bench.s3.amazonaws.com/snli/snli_test.csv', delimiter="|")
```

## Training

AutoMM uses BERT to project sentences into high-dimensional vectors and treats matching as classification following [sentence transformers](https://www.sbert.net/) design.

**Key Configuration:**
- Labels must be **binary**
- Define `match_label` to indicate semantic equivalence (context-dependent)

```python
from autogluon.multimodal import MultiModalPredictor

predictor = MultiModalPredictor(
    problem_type="text_similarity",
    query="premise",           # first sentence column
    response="hypothesis",     # second sentence column
    label="label",            # label column
    match_label=1,            # label for semantic match
    eval_metric='auc',
)

predictor.fit(train_data=snli_train, time_limit=180)
```

## Evaluation & Prediction

```python
# Evaluate
score = predictor.evaluate(snli_test)

# Predict match (binary)
pred_data = pd.DataFrame.from_dict({
    "premise":["The teacher gave his speech to an empty room."], 
    "hypothesis":["There was almost nobody when the professor was talking."]
})
predictions = predictor.predict(pred_data)

# Predict probabilities
probabilities = predictor.predict_proba(pred_data)

# Extract embeddings separately
embeddings_1 = predictor.extract_embedding({"premise":["..."]})
embeddings_2 = predictor.extract_embedding({"hypothesis":["..."]})
```

## Additional Resources
- [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm)
- [Customize AutoMM](../advanced_topics/customization.ipynb)