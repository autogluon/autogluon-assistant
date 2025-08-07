# Condensed: ```python

Summary: This tutorial demonstrates how to create and use custom evaluation metrics in AutoGluon, focusing on what makes them better for model optimization. It covers the implementation of custom metrics for classification, regression, and probability-based tasks, with examples of accuracy, ROC AUC, and MSE. The metrics are defined using make_scorer and can be used for model evaluation and comparison.

*This is a condensed version that preserves essential implementation details and context.*

# Custom Metrics in AutoGluon

## Setup
```python
!pip install autogluon.tabular[all]
import numpy as np
from autogluon.core.metrics import make_scorer
import sklearn.metrics
```

## Important: Ensuring Metric is Serializable
Custom metrics must be defined in a separate Python file and imported to be pickled properly. If not serializable, you'll get `_pickle.PicklingError: Can't pickle` errors when using parallel training.

**Best Practice:** Create a file like `my_metrics.py` with your metric definitions, then import with `from my_metrics import ag_accuracy_scorer`.

## Creating Custom Metrics

### Custom Accuracy Metric
```python
ag_accuracy_scorer = make_scorer(
    name='accuracy',
    score_func=sklearn.metrics.accuracy_score,
    optimum=1,
    greater_is_better=True,
    needs_class=True
)
```

### Custom Mean Squared Error Metric
```python
ag_mean_squared_error_scorer = make_scorer(
    name='mean_squared_error',
    score_func=sklearn.metrics.mean_squared_error,
    optimum=0,
    greater_is_better=False
)
```

### Custom Function Example
```python
def mse_func(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return ((y_true - y_pred) ** 2).mean()

ag_mean_squared_error_custom_scorer = make_scorer(
    name='mean_squared_error',
    score_func=mse_func,
    optimum=0,
    greater_is_better=False
)
```

### Custom ROC AUC Metric
```python
ag_roc_auc_scorer = make_scorer(
    name='roc_auc',
    score_func=sklearn.metrics.roc_auc_score,
    optimum=1,
    greater_is_better=True,
    needs_threshold=True
)
```

## Key Parameters for `make_scorer`
- `name`: Name for the scorer (used in logs)
- `score_func`: The function to wrap
- `optimum`: Best possible value from the original metric
- `greater_is_better`: Whether higher values are better
- `needs_*`: Specify the type of predictions required:
  - `needs_pred`: For regression metrics (default if no others specified)
  - `needs_proba`: For probability estimates
  - `needs_class`: For classification predictions
  - `needs_threshold`: For continuous decision certainty (binary classification)
  - `needs_quantile`: For quantile regression

## Using Custom Metrics in TabularPredictor

### With leaderboard
```python
predictor = TabularPredictor(label=label).fit(train_data, hyperparameters='toy')
predictor.leaderboard(test_data, extra_metrics=[ag_roc_auc_scorer, ag_accuracy_scorer])
```

### As evaluation metric
```python
predictor_custom = TabularPredictor(label=label, eval_metric=ag_roc_auc_scorer).fit(train_data, hyperparameters='toy')
predictor_custom.leaderboard(test_data)
```

## Important Notes
- AutoGluon Scorers always report scores in `greater_is_better=True` form internally
- For metrics where `greater_is_better=False`, AutoGluon flips the value
- Error is calculated as `sign * optimum - score` where `sign=1` if `greater_is_better=True`, else `sign=-1`
- Error is always in `lower_is_better` format (0 = perfect prediction)