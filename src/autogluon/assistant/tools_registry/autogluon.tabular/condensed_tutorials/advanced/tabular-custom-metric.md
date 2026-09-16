# Condensed: ```python

Summary: This tutorial teaches how to create and integrate custom evaluation metrics in AutoGluon using `make_scorer`. It covers wrapping sklearn or custom metric functions with proper configuration of key parameters: `optimum` (theoretical best value), `greater_is_better` (optimization direction), and `needs_*` flags (needs_class for classification, needs_proba for probabilities, needs_threshold for ranking metrics, needs_pred for regression). Critical implementation detail: metrics must be defined in separate Python files for pickle serialization to avoid Ray parallelization errors. Includes complete examples for accuracy (classification), MSE (regression), and ROC AUC (threshold) metrics, plus integration with TabularPredictor via `eval_metric` parameter or `extra_metrics` in leaderboard. Provides utility methods for score/error conversion and emphasizes correct parameter configuration to prevent optimizing for worst models.

*This is a condensed version that preserves essential implementation details and context.*

# Custom Metrics in AutoGluon - Condensed Tutorial

## Installation
```python
!pip install autogluon.tabular[all]
```

## Critical Requirement: Metric Serialization

**⚠️ IMPORTANT**: Custom metrics must be defined in a separate Python file and imported to be pickleable for Ray parallelization.

```python
# Create my_metrics.py with your metric, then:
from my_metrics import ag_accuracy_scorer
```

**Error indicator**: `_pickle.PicklingError: Can't pickle` means your metric isn't serializable.

## Creating Custom Scorers with `make_scorer`

### Basic Structure
```python
from autogluon.core.metrics import make_scorer

ag_custom_scorer = make_scorer(
    name='metric_name',
    score_func=your_metric_function,
    optimum=optimal_value,
    greater_is_better=True/False,
    needs_class=True  # or needs_proba, needs_threshold, needs_quantile
)
```

### Key Parameters

**`needs_*` arguments** (only one can be True):
- `needs_pred`: Regression metrics (default if others are False) - e.g., RMSE, MAE
- `needs_proba`: Requires probability estimates - e.g., log_loss, roc_auc_ovo
- `needs_class`: Requires class predictions - e.g., accuracy, F1, precision
- `needs_threshold`: Binary classification decision certainty - e.g., roc_auc, average_precision
- `needs_quantile`: Quantile regression - e.g., pinball_loss

**Score conversion**: AutoGluon always converts scores to `greater_is_better=True` format internally.
- `error = sign * optimum - score` (where `sign=1` if `greater_is_better=True`, else `-1`)

## Example 1: Custom Accuracy Metric

```python
import sklearn.metrics

ag_accuracy_scorer = make_scorer(
    name='accuracy',
    score_func=sklearn.metrics.accuracy_score,
    optimum=1,
    greater_is_better=True,
    needs_class=True
)

# Usage
score = ag_accuracy_scorer(y_true, y_pred)
error = ag_accuracy_scorer.error(y_true, y_pred)  # error = 1 - score
```

## Example 2: Custom MSE Metric (Regression)

```python
ag_mse_scorer = make_scorer(
    name='mean_squared_error',
    score_func=sklearn.metrics.mean_squared_error,
    optimum=0,
    greater_is_better=False
)

# Returns negative value (converted to greater_is_better format)
score = ag_mse_scorer(y_true, y_pred)
```

**Custom implementation**:
```python
def mse_func(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return ((y_true - y_pred) ** 2).mean()

ag_mse_custom = make_scorer(
    name='mse',
    score_func=mse_func,
    optimum=0,
    greater_is_better=False
)
```

## Example 3: Custom ROC AUC (Threshold Metric)

```python
ag_roc_auc_scorer = make_scorer(
    name='roc_auc',
    score_func=sklearn.metrics.roc_auc_score,
    optimum=1,
    greater_is_better=True,
    needs_threshold=True  # Cares about rank order
)
```

## Using Custom Metrics in TabularPredictor

### As Evaluation Metric
```python
from autogluon.tabular import TabularPredictor

predictor = TabularPredictor(
    label=label, 
    eval_metric=ag_roc_auc_scorer
).fit(train_data, hyperparameters='toy')
```

### In Leaderboard
```python
# Add extra metrics to leaderboard
predictor.leaderboard(
    test_data, 
    extra_metrics=[ag_roc_auc_scorer, ag_accuracy_scorer]
)
```

## Best Practices

1. **Always define custom metrics in separate files** for serialization
2. **Set `greater_is_better` correctly** or AutoGluon will optimize for worst model
3. **Specify correct `needs_*` parameter** based on metric type
4. **Set `optimum` to the theoretical best value** of the original metric
5. Metric functions must accept `(y_true, y_pred)` as numpy arrays and return float

## Utility Methods
```python
# Convert between score and error
error = ag_scorer.convert_score_to_error(score)
score = ag_scorer.convert_error_to_score(error)

# Get original metric value
score_orig = ag_scorer.convert_score_to_original(score)
```