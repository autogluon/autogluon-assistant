# Condensed: ```python

Summary: This tutorial demonstrates AutoGluon's MultiModalPredictor for text classification using different quality presets. It covers implementation techniques for sentiment analysis with three preset configurations: medium_quality (fast, smaller models), high_quality (balanced), and best_quality (performance-focused). The tutorial shows how to add hyperparameter optimization with _hpo suffixes, evaluate models using metrics like roc_auc, and view preset configurations. Key functionalities include automatic model selection, time-limited training, and hyperparameter tuning for text classification tasks with minimal code, making it useful for quickly implementing sentiment analysis or other text classification problems.

*This is a condensed version that preserves essential implementation details and context.*

# AutoMM Presets Tutorial

## Setup
```python
!pip install autogluon.multimodal
import warnings
warnings.filterwarnings('ignore')
```

## Dataset
Using Stanford Sentiment Treebank (SST) for binary classification of movie reviews.

```python
from autogluon.core.utils.loaders import load_pd

train_data = load_pd.load('https://autogluon-text.s3-accelerate.amazonaws.com/glue/sst/train.parquet')
test_data = load_pd.load('https://autogluon-text.s3-accelerate.amazonaws.com/glue/sst/dev.parquet')
subsample_size = 1000  # subsample for faster demo
train_data = train_data.sample(n=subsample_size, random_state=0)
```

## Quality Presets

### Medium Quality
Optimized for fast training and inference with smaller models.

```python
from autogluon.multimodal import MultiModalPredictor

predictor = MultiModalPredictor(label='label', eval_metric='acc', presets="medium_quality")
predictor.fit(train_data=train_data, time_limit=20)  # seconds
scores = predictor.evaluate(test_data, metrics=["roc_auc"])
```

### High Quality
Balances prediction quality and speed using larger models.

```python
predictor = MultiModalPredictor(label='label', eval_metric='acc', presets="high_quality")
predictor.fit(train_data=train_data, time_limit=20)
scores = predictor.evaluate(test_data, metrics=["roc_auc"])
```

### Best Quality
Prioritizes performance over training/inference cost. Requires more time and GPU memory.

```python
predictor = MultiModalPredictor(label='label', eval_metric='acc', presets="best_quality")
predictor.fit(train_data=train_data, time_limit=180)
scores = predictor.evaluate(test_data, metrics=["roc_auc"])
```

## HPO Presets
Add `_hpo` suffix to enable hyperparameter optimization:
- `medium_quality_hpo`
- `high_quality_hpo`
- `best_quality_hpo`

## Viewing Preset Configurations
```python
import json
from autogluon.multimodal.utils.presets import get_presets

# View high_quality preset details
hyperparameters, hyperparameter_tune_kwargs = get_presets(problem_type="default", presets="high_quality")
print(f"hyperparameters: {json.dumps(hyperparameters, sort_keys=True, indent=4)}")

# View high_quality_hpo preset details
hyperparameters, hyperparameter_tune_kwargs = get_presets(problem_type="default", presets="high_quality_hpo")
```

HPO presets tune parameters like model backbone, batch size, learning rate, max epoch, and optimizer type.

## Additional Resources
- More examples: [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm)
- Customization: Refer to "Customize AutoMM" documentation