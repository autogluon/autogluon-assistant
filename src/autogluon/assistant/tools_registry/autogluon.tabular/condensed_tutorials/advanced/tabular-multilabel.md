# Condensed: Predicting Multiple Columns in a Table (Multi-Label Prediction)

Summary: This tutorial covers AutoGluon's MultilabelPredictor implementation for handling multiple prediction tasks simultaneously. It demonstrates how to build models that can predict different types of targets (regression, classification) while considering label correlations. Key implementation knowledge includes initializing the predictor with different problem types and metrics, training with time limits, and accessing individual predictors. The tutorial helps with tasks involving multi-target prediction, model optimization, and memory management. Notable features include label correlation handling, support for mixed problem types (regression/classification), performance optimization through presets, and model persistence capabilities. It's particularly useful for developers working on complex prediction tasks requiring multiple interdependent outputs.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details:

# Multi-Label Prediction in AutoGluon

## Key Implementation Details

### MultilabelPredictor Class
```python
class MultilabelPredictor:
    def __init__(self, labels, path=None, problem_types=None, eval_metrics=None, 
                 consider_labels_correlation=True, **kwargs):
        # Core parameters:
        # - labels: List of columns to predict
        # - problem_types: List of prediction types for each label
        # - eval_metrics: List of metrics for each label
        # - consider_labels_correlation: Whether to account for label dependencies
```

### Critical Configurations
```python
# Example setup
labels = ['education-num', 'education', 'class']
problem_types = ['regression', 'multiclass', 'binary']
eval_metrics = ['mean_absolute_error', 'accuracy', 'accuracy']
save_path = 'agModels-predictEducationClass'
```

### Basic Usage
```python
# Initialize
multi_predictor = MultilabelPredictor(
    labels=labels, 
    problem_types=problem_types, 
    eval_metrics=eval_metrics, 
    path=save_path
)

# Train
multi_predictor.fit(train_data, time_limit=time_limit)

# Predict
predictions = multi_predictor.predict(test_data)

# Evaluate
evaluations = multi_predictor.evaluate(test_data)
```

## Best Practices

1. **Performance Optimization**:
   - Set `presets='best_quality'` for optimal predictions
   - Use `consider_labels_correlation=False` if planning to use individual predictors

2. **Memory Management**:
   - For memory issues: Use strategies from tabular-indepth tutorial
   - For faster inference: Use preset `['good_quality', 'optimize_for_deployment']`

3. **Model Access**:
   ```python
   # Access individual predictor
   predictor_specific = multi_predictor.get_predictor('label_name')
   ```

## Important Notes

- Requires at least 2 labels for prediction
- Saves separate TabularPredictor for each label
- Label correlation handling depends on order in labels list
- Can load/save models using `load()` and `save()` methods

## Warning

When `consider_labels_correlation=True`, prediction order matters as each label is predicted conditionally on previous labels in the sequence.