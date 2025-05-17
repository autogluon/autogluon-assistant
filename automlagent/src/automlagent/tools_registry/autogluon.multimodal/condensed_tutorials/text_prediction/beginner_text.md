# Condensed: AutoMM for Text - Quick Start

Summary: This tutorial demonstrates implementing text analysis tasks using AutoGluon's MultiModalPredictor, specifically focusing on sentiment analysis and sentence similarity. It provides code examples for model training, evaluation, prediction, and model management (save/load operations). Key implementation knowledge includes setting up the predictor with proper configurations (label columns, evaluation metrics, time limits), handling data in table format, and extracting embeddings. The tutorial helps with tasks like training text classifiers, generating predictions and probabilities, and managing trained models. Notable features covered include support for multiple text columns, integration with popular ML libraries (timm, huggingface, CLIP), and various evaluation metrics for both classification and regression tasks.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details:

# AutoMM for Text - Quick Start

## Key Components
- `MultiModalPredictor` handles text, image, numerical, and categorical data
- Demonstrates sentiment analysis and sentence similarity tasks
- Uses data tables with text features and label columns

## Implementation Details

### Setup
```python
pip install autogluon.multimodal
from autogluon.multimodal import MultiModalPredictor
```

### Sentiment Analysis Implementation

1. **Training**
```python
predictor = MultiModalPredictor(
    label='label',
    eval_metric='acc',
    path='model_path'
)
predictor.fit(train_data, time_limit=180)
```

2. **Evaluation & Prediction**
```python
# Evaluate with multiple metrics
test_score = predictor.evaluate(test_data, metrics=['acc', 'f1'])

# Single prediction
predictions = predictor.predict({'sentence': [text]})

# Probability predictions
probs = predictor.predict_proba({'sentence': [text]})
```

3. **Model Management**
```python
# Save model
predictor.save(model_path)

# Load model
loaded_predictor = MultiModalPredictor.load(model_path)
```

### Sentence Similarity Implementation

```python
predictor_sts = MultiModalPredictor(
    label='score',
    path='sts_model_path'
)
predictor_sts.fit(sts_train_data, time_limit=60)
```

## Critical Configurations

- **Label Column**: Specify using `label` parameter
- **Evaluation Metrics**: 
  - Classification: 'acc', 'f1'
  - Regression: 'rmse', 'pearsonr', 'spearmanr'
- **Time Limit**: Set longer for better performance (recommended: 1+ hour)

## Best Practices

1. Use adequate training time (>1 hour) for production models
2. Include multiple evaluation metrics for comprehensive assessment
3. Format data as tables with clear feature and label columns
4. For text tasks, ensure proper data formatting in table structure

## Important Warnings

- `MultiModalPredictor.load()` uses pickle - only load trusted data
- Short time limits (< 180s) are for demonstration only
- Subsample size affects model performance

## Advanced Features

- Extract embeddings:
```python
embeddings = predictor.extract_embedding(data)
```
- Supports multiple text columns in data tables
- Integrates with timm, huggingface/transformers, and openai/clip