# Condensed: ```python

Summary: This tutorial demonstrates using AutoGluon MultiModal for sentiment analysis and sentence similarity tasks. It covers implementation techniques for training models with MultiModalPredictor, evaluating performance with multiple metrics, making predictions on text data, extracting embeddings for visualization, and saving/loading models. The tutorial helps with text classification, regression for semantic similarity, and embedding extraction tasks. Key features include automatic problem type detection, fine-tuning of deep learning models, integration with transformer models, customizable evaluation metrics, and visualization of embeddings using TSNE. The code provides a complete workflow from data loading to model deployment for NLP tasks.

*This is a condensed version that preserves essential implementation details and context.*

# Sentiment Analysis and Sentence Similarity with AutoGluon MultiModal

## Setup

```python
!pip install autogluon.multimodal

import numpy as np
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
np.random.seed(123)
```

## Sentiment Analysis Task

### Loading Data

```python
from autogluon.core.utils.loaders import load_pd
train_data = load_pd.load('https://autogluon-text.s3-accelerate.amazonaws.com/glue/sst/train.parquet')
test_data = load_pd.load('https://autogluon-text.s3-accelerate.amazonaws.com/glue/sst/dev.parquet')
subsample_size = 1000  # subsample data for faster demo
train_data = train_data.sample(n=subsample_size, random_state=0)
```

### Training

```python
from autogluon.multimodal import MultiModalPredictor
import uuid
model_path = f"./tmp/{uuid.uuid4().hex}-automm_sst"
predictor = MultiModalPredictor(label='label', eval_metric='acc', path=model_path)
predictor.fit(train_data, time_limit=180)  # For better performance, increase time_limit or set to None
```

### Evaluation

```python
# Basic evaluation
test_score = predictor.evaluate(test_data)
print(test_score)

# Multiple metrics
test_score = predictor.evaluate(test_data, metrics=['acc', 'f1'])
print(test_score)
```

### Prediction

```python
# Predict individual sentences
sentence1 = "it's a charming and often affecting journey."
sentence2 = "It's slow, very, very, very slow."
predictions = predictor.predict({'sentence': [sentence1, sentence2]})

# Get class probabilities
probs = predictor.predict_proba({'sentence': [sentence1, sentence2]})

# Predict on dataset
test_predictions = predictor.predict(test_data)
```

### Save and Load

```python
# Load saved predictor
loaded_predictor = MultiModalPredictor.load(model_path)

# Save to new location
new_model_path = f"./tmp/{uuid.uuid4().hex}-automm_sst"
loaded_predictor.save(new_model_path)
```

> **Warning**: `MultiModalPredictor.load()` uses `pickle` module, which can execute arbitrary code during unpickling. Only load data you trust.

### Extract Embeddings

```python
embeddings = predictor.extract_embedding(test_data)
print(embeddings.shape)

# Visualize embeddings with TSNE
from sklearn.manifold import TSNE
X_embedded = TSNE(n_components=2, random_state=123).fit_transform(embeddings)
for val, color in [(0, 'red'), (1, 'blue')]:
    idx = (test_data['label'].to_numpy() == val).nonzero()
    plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], c=color, label=f'label={val}')
plt.legend(loc='best')
```

## Sentence Similarity Task

### Loading Data

```python
sts_train_data = load_pd.load('https://autogluon-text.s3-accelerate.amazonaws.com/glue/sts/train.parquet')[['sentence1', 'sentence2', 'score']]
sts_test_data = load_pd.load('https://autogluon-text.s3-accelerate.amazonaws.com/glue/sts/dev.parquet')[['sentence1', 'sentence2', 'score']]
print('Min score=', min(sts_train_data['score']), ', Max score=', max(sts_train_data['score']))
```

### Training and Evaluation

```python
sts_model_path = f"./tmp/{uuid.uuid4().hex}-automm_sts"
predictor_sts = MultiModalPredictor(label='score', path=sts_model_path)
predictor_sts.fit(sts_train_data, time_limit=60)

test_score = predictor_sts.evaluate(sts_test_data, metrics=['rmse', 'pearsonr', 'spearmanr'])
print('RMSE = {:.2f}'.format(test_score['rmse']))
print('PEARSONR = {:.4f}'.format(test_score['pearsonr']))
print('SPEARMANR = {:.4f}'.format(test_score['spearmanr']))
```

### Prediction

```python
sentences = ['The child is riding a horse.',
             'The young boy is riding a horse.',
             'The young man is riding a horse.',
             'The young man is riding a bicycle.']

score1 = predictor_sts.predict({'sentence1': [sentences[0]],
                                'sentence2': [sentences[1]]}, as_pandas=False)
score2 = predictor_sts.predict({'sentence1': [sentences[0]],
                                'sentence2': [sentences[2]]}, as_pandas=False)
score3 = predictor_sts.predict({'sentence1': [sentences[0]],
                                'sentence2': [sentences[3]]}, as_pandas=False)
```

## Key Notes

- `MultiModalPredictor` automatically determines the type of prediction problem and appropriate loss function
- Unlike `TabularPredictor`, `MultiModalPredictor` focuses on selecting and fine-tuning deep learning models
- Internally integrates with timm, huggingface/transformers, and openai/clip as the model zoo
- For customization, refer to the "Customize AutoMM" documentation