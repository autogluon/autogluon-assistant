# Condensed: Adding a custom metric to AutoGluon

Summary: This tutorial demonstrates how to implement custom evaluation metrics in AutoGluon using the make_scorer() function. It covers the technical implementation of creating serializable custom metrics for different types of machine learning tasks (classification, regression, probability-based) through detailed examples. The tutorial helps with tasks like defining custom accuracy, MSE, and ROC AUC metrics, integrating them into model training and evaluation workflows. Key features include the essential parameters for make_scorer(), proper metric serialization requirements, handling different prediction types (class, probability, threshold-based), and best practices for implementing custom metric functions that are compatible with AutoGluon's framework.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details:

# Adding Custom Metrics to AutoGluon

## Key Implementation Details

### Creating a Custom Metric
Custom metrics must be defined in a separate Python file and imported to ensure they are serializable (pickleable).

```python
from autogluon.core.metrics import make_scorer

# Basic structure for creating a custom scorer
custom_scorer = make_scorer(
    name='metric_name',
    score_func=function_that_calculates_metric,
    optimum=optimal_value,
    greater_is_better=True/False,
    needs_pred/needs_proba/needs_class/needs_threshold/needs_quantile=True
)
```

### Critical Parameters for make_scorer()

- `name`: Identifier for the scorer
- `score_func`: Function that calculates the metric
- `optimum`: Best possible value for the metric
- `greater_is_better`: Whether higher scores are better
- `needs_*`: Specify type of predictions required:
  - `needs_pred`: For regression metrics
  - `needs_proba`: For probability estimates
  - `needs_class`: For classification predictions
  - `needs_threshold`: For binary classification metrics using decision certainty
  - `needs_quantile`: For quantile regression metrics

## Implementation Examples

### 1. Custom Accuracy Metric
```python
import sklearn.metrics

ag_accuracy_scorer = make_scorer(
    name='accuracy',
    score_func=sklearn.metrics.accuracy_score,
    optimum=1,
    greater_is_better=True,
    needs_class=True
)
```

### 2. Custom Mean Squared Error
```python
ag_mse_scorer = make_scorer(
    name='mean_squared_error',
    score_func=sklearn.metrics.mean_squared_error,
    optimum=0,
    greater_is_better=False
)
```

### 3. Custom ROC AUC
```python
ag_roc_auc_scorer = make_scorer(
    name='roc_auc',
    score_func=sklearn.metrics.roc_auc_score,
    optimum=1,
    greater_is_better=True,
    needs_threshold=True
)
```

## Using Custom Metrics

### With Leaderboard
```python
predictor.leaderboard(test_data, extra_metrics=[custom_scorer1, custom_scorer2])
```

### During Training
```python
predictor = TabularPredictor(
    label=label, 
    eval_metric=custom_scorer
).fit(train_data)
```

## Important Warnings and Best Practices

1. Custom metrics must be defined in separate Python files to be serializable
2. Non-serializable metrics will crash during training with `_pickle.PicklingError`
3. Ensure `greater_is_better` is set correctly to avoid optimizing for worst models
4. AutoGluon Scorers internally convert all metrics to `greater_is_better=True` format
5. Custom metric functions must accept `y_true` and `y_pred` as numpy arrays and return a float