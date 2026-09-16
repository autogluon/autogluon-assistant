# Condensed: ```python

Summary: This tutorial demonstrates AutoGluon MultiModalPredictor implementation for text classification tasks using three quality presets: `medium_quality` (fast, small models), `high_quality` (balanced performance), and `best_quality` (maximum accuracy, GPU-intensive). It covers essential coding patterns including data loading with `load_pd`, model initialization with `MultiModalPredictor(label, eval_metric, presets)`, training with `fit(train_data, time_limit)`, and evaluation with `evaluate(test_data, metrics)`. Key features include HPO-enabled presets (add `_hpo` postfix) for automatic hyperparameter tuning of model backbone, batch size, learning rate, and optimizer, plus utilities to inspect preset configurations using `get_presets()`. Useful for implementing sentiment analysis, binary classification, and multimodal ML pipelines with configurable speed-accuracy tradeoffs.

*This is a condensed version that preserves essential implementation details and context.*

# AutoGluon MultiModal Presets Tutorial

## Installation & Setup
```python
!pip install autogluon.multimodal
import warnings
warnings.filterwarnings('ignore')
```

## Dataset
Uses Stanford Sentiment Treebank (SST) for binary sentiment classification (1=positive, 0=negative).

```python
from autogluon.core.utils.loaders import load_pd

train_data = load_pd.load('https://autogluon-text.s3-accelerate.amazonaws.com/glue/sst/train.parquet')
test_data = load_pd.load('https://autogluon-text.s3-accelerate.amazonaws.com/glue/sst/dev.parquet')
subsample_size = 1000  # subsample for faster demo
train_data = train_data.sample(n=subsample_size, random_state=0)
```

## Quality Presets

### 1. Medium Quality (Fast Training/Inference)
Smallest model size, fastest training.

```python
from autogluon.multimodal import MultiModalPredictor

predictor = MultiModalPredictor(label='label', eval_metric='acc', presets="medium_quality")
predictor.fit(train_data=train_data, time_limit=20)
scores = predictor.evaluate(test_data, metrics=["roc_auc"])
```

### 2. High Quality (Balanced)
Larger model, better performance, requires more time.

```python
predictor = MultiModalPredictor(label='label', eval_metric='acc', presets="high_quality")
predictor.fit(train_data=train_data, time_limit=20)
```

### 3. Best Quality (Maximum Performance)
Best performance, highest cost. **Requires high-end GPUs with large memory.**

```python
predictor = MultiModalPredictor(label='label', eval_metric='acc', presets="best_quality")
predictor.fit(train_data=train_data, time_limit=180)
```

## HPO Presets
Add `_hpo` postfix for hyperparameter optimization: `medium_quality_hpo`, `high_quality_hpo`, `best_quality_hpo`.

HPO tunes: model backbone, batch size, learning rate, max epoch, optimizer type.

## Inspect Preset Details

```python
import json
from autogluon.multimodal.utils.presets import get_presets

hyperparameters, hyperparameter_tune_kwargs = get_presets(problem_type="default", presets="high_quality")
print(f"hyperparameters: {json.dumps(hyperparameters, sort_keys=True, indent=4)}")
```

For HPO presets with YAML output:
```python
import yaml
hyperparameters, hyperparameter_tune_kwargs = get_presets(problem_type="default", presets="high_quality_hpo")
print(f"hyperparameters: {yaml.dump(hyperparameters, allow_unicode=True, default_flow_style=False)}")
```

## Additional Resources
- **Examples**: [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm)
- **Customization**: [Customize AutoMM](../advanced_topics/customization.ipynb)