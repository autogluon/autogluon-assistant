# Condensed: ```python

Summary: This tutorial covers AutoGluon's MultiModalPredictor for automated NLP tasks including text classification and regression. It demonstrates loading data from Parquet/CSV/S3, training models with `fit()` using configurable parameters (label, eval_metric, time_limit), making predictions with `predict()` and `predict_proba()`, evaluating with custom metrics, and extracting neural network embeddings. Key implementations include binary sentiment classification, sentence similarity regression with multi-column inputs, model persistence via save/load (with pickle security warnings), and automatic task detection. The tutorial helps with building text classifiers, regression models for semantic similarity, embedding extraction for downstream tasks, and handling tabular text data with multiple columns, leveraging HuggingFace Transformers, timm, and CLIP backends for deep learning-based NLP solutions.

*This is a condensed version that preserves essential implementation details and context.*

# AutoGluon MultiModal Tutorial (Condensed)

## Installation & Setup
```python
!pip install autogluon.multimodal
```

## Sentiment Analysis (Binary Classification)

### Data Loading
```python
from autogluon.core.utils.loaders import load_pd
train_data = load_pd.load('https://autogluon-text.s3-accelerate.amazonaws.com/glue/sst/train.parquet')
test_data = load_pd.load('https://autogluon-text.s3-accelerate.amazonaws.com/glue/sst/dev.parquet')
subsample_size = 1000  # increase for better performance
train_data = train_data.sample(n=subsample_size, random_state=0)
```

**Data Format**: Pandas DataFrame with label column. Supports Parquet, CSV, and local/S3 files.

### Training
```python
from autogluon.multimodal import MultiModalPredictor
predictor = MultiModalPredictor(label='label', eval_metric='acc', path=model_path)
predictor.fit(train_data, time_limit=180)
```

**Key Parameters**:
- `label`: Column containing target values
- `eval_metric`: Optimization metric (e.g., 'acc', 'f1')
- `time_limit`: Training duration in seconds (use longer times or `None` for production)

### Evaluation & Prediction
```python
# Evaluation with multiple metrics
test_score = predictor.evaluate(test_data, metrics=['acc', 'f1'])

# Predictions
predictions = predictor.predict({'sentence': [sentence1, sentence2]})

# Class probabilities
probs = predictor.predict_proba({'sentence': [sentence1, sentence2]})
```

### Save & Load

⚠️ **Security Warning**: `MultiModalPredictor.load()` uses `pickle`, which is insecure. Only load data from trusted sources.

```python
# Auto-saved after fit(), reload with:
loaded_predictor = MultiModalPredictor.load(model_path)

# Manual save
loaded_predictor.save(new_model_path)
```

### Extract Embeddings
```python
embeddings = predictor.extract_embedding(test_data)
# Returns embedding vectors from intermediate neural network representations
```

## Sentence Similarity (Regression)

### Data & Training
```python
sts_train_data = load_pd.load('https://autogluon-text.s3-accelerate.amazonaws.com/glue/sts/train.parquet')[['sentence1', 'sentence2', 'score']]

predictor_sts = MultiModalPredictor(label='score', path=sts_model_path)
predictor_sts.fit(sts_train_data, time_limit=60)
```

**Note**: AutoGluon automatically detects regression vs classification based on label column.

### Evaluation with Custom Metrics
```python
test_score = predictor_sts.evaluate(sts_test_data, metrics=['rmse', 'pearsonr', 'spearmanr'])
```

### Multi-Column Prediction
```python
score = predictor_sts.predict({'sentence1': [sentences[0]],
                               'sentence2': [sentences[1]]}, as_pandas=False)
```

## Key Concepts

- **Automatic Task Detection**: Determines classification/regression from data
- **Multi-Column Support**: Handles multiple text columns in data tables
- **Model Backend**: Integrates timm, HuggingFace Transformers, OpenAI CLIP
- **Focus**: Deep learning model selection and finetuning (unlike TabularPredictor's ensemble approach)

## Best Practices

1. **Time Limits**: Use 1+ hour for production; short limits are for demos only
2. **Data Format**: Any tabular format with proper label column works
3. **Metrics**: Specify multiple metrics during evaluation for comprehensive assessment
4. **Security**: Never load models from untrusted sources

## Additional Resources
- [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm)
- [Customization Guide](../advanced_topics/customization.ipynb)