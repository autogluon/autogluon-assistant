# Condensed: ```python

Summary: This tutorial demonstrates AutoGluon TabularPredictor for machine learning tasks, covering hyperparameter tuning with search spaces for neural networks and gradient boosting models, model ensembling through stacking/bagging, and decision threshold calibration for binary classification. It explains inference optimization techniques including model persistence, inference speed constraints, ensemble reduction, and model distillation. The tutorial provides practical code examples for accelerating predictions (up to 160x speedup), managing memory usage, and evaluating model performance. Key functionalities include feature importance analysis, loading/saving predictors, making batch and single-instance predictions, and optimizing deployment with techniques like refit_full, persist, and infer_limit parameters.

*This is a condensed version that preserves essential implementation details and context.*

# AutoGluon TabularPredictor Tutorial: Hyperparameter Tuning and Model Ensembling

## Setup and Data Loading

```python
!pip install autogluon.tabular[all]

from autogluon.tabular import TabularDataset, TabularPredictor
import numpy as np

# Load and sample data
train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
train_data = train_data.sample(n=1000, random_state=0)  # subsample for faster demo

label = 'occupation'
test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')
test_data_nolabel = test_data.drop(columns=[label])

metric = 'accuracy'  # evaluation metric
```

## Hyperparameter Tuning

> **Note: AutoGluon typically achieves best performance without hyperparameter tuning by simply using `presets="best_quality"`**

```python
from autogluon.common import space

# Define hyperparameter search spaces
nn_options = {
    'num_epochs': 10,
    'learning_rate': space.Real(1e-4, 1e-2, default=5e-4, log=True),
    'activation': space.Categorical('relu', 'softrelu', 'tanh'),
    'dropout_prob': space.Real(0.0, 0.5, default=0.1),
}

gbm_options = {
    'num_boost_round': 100,
    'num_leaves': space.Int(lower=26, upper=66, default=36),
}

hyperparameters = {
    'GBM': gbm_options,
    'NN_TORCH': nn_options,  # Comment out if errors on Mac OSX
}

# HPO configuration
hyperparameter_tune_kwargs = {
    'num_trials': 5,
    'scheduler': 'local',
    'searcher': 'auto',
}

# Train with hyperparameter tuning
predictor = TabularPredictor(label=label, eval_metric=metric).fit(
    train_data,
    time_limit=2*60,  # 2 minutes
    hyperparameters=hyperparameters,
    hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
)

# Predict and evaluate
y_pred = predictor.predict(test_data_nolabel)
perf = predictor.evaluate(test_data, auxiliary_metrics=False)

# View training summary
results = predictor.fit_summary()
```

## Model Ensembling with Stacking/Bagging

```python
# Switch to binary classification task
label = 'class'
test_data_nolabel = test_data.drop(columns=[label])
y_test = test_data[label]
save_path = 'agModels-predictClass'

# Train with bagging and stacking
predictor = TabularPredictor(label=label, eval_metric=metric).fit(
    train_data,
    num_bag_folds=5,     # k-fold bagging
    num_bag_sets=1,      # number of bagging iterations
    num_stack_levels=1,  # number of stacking levels
    # Reduced hyperparameters for quick demo only
    hyperparameters={'NN_TORCH': {'num_epochs': 2}, 'GBM': {'num_boost_round': 20}},
)
```

**Important Notes:**
- Don't provide `tuning_data` when using stacking/bagging - AutoGluon will intelligently split the data
- Increasing `num_bag_sets` may improve accuracy but significantly increases training time and resource usage
- Use `auto_stack` parameter (part of `best_quality` preset) to let AutoGluon automatically select optimal stacking/bagging values

# Decision Threshold Calibration in TabularPredictor

## Initial Setup and Training

```python
predictor = TabularPredictor(label=label, eval_metric='balanced_accuracy', path=save_path).fit(
    train_data, auto_stack=True,
    calibrate_decision_threshold=False,  # Disabled for demonstration
    hyperparameters={'FASTAI': {'num_epochs': 10}, 'GBM': {'num_boost_round': 200}}  # For quick demo only
)
predictor.leaderboard(test_data)
```

> **Note**: Stacking/bagging often produces better accuracy than hyperparameter-tuning alone. Consider using `presets='best_quality'` which sets `auto_stack=True`.

## Decision Threshold Calibration

For binary classification, adjusting the prediction threshold can significantly improve metrics like `f1` and `balanced_accuracy`.

### Basic Calibration Example

```python
# Evaluate before calibration
print(f'Prior to calibration (predictor.decision_threshold={predictor.decision_threshold}):')
scores = predictor.evaluate(test_data)

# Calibrate and set new threshold
calibrated_decision_threshold = predictor.calibrate_decision_threshold()
predictor.set_decision_threshold(calibrated_decision_threshold)

# Evaluate after calibration
print(f'After calibration (predictor.decision_threshold={predictor.decision_threshold}):')
scores_calibrated = predictor.evaluate(test_data)
```

### Calibration Trade-offs

Calibrating for one metric (like "balanced_accuracy") may improve that metric but harm others (like "accuracy"). This represents a performance trade-off between different metrics.

### Calibrating for Specific Metrics

```python
predictor.set_decision_threshold(0.5)  # Reset threshold
for metric_name in ['f1', 'balanced_accuracy', 'mcc']:
    # Get baseline score
    metric_score = predictor.evaluate(test_data, silent=True)[metric_name]
    
    # Calibrate for specific metric
    calibrated_decision_threshold = predictor.calibrate_decision_threshold(metric=metric_name, verbose=False)
    
    # Evaluate with calibrated threshold
    metric_score_calibrated = predictor.evaluate(
        test_data, decision_threshold=calibrated_decision_threshold, silent=True
    )[metric_name]
    
    print(f'decision_threshold={calibrated_decision_threshold:.3f}\t| metric="{metric_name}"'
          f'\n\ttest_score uncalibrated: {metric_score:.4f}'
          f'\n\ttest_score   calibrated: {metric_score_calibrated:.4f}'
          f'\n\ttest_score        delta: {metric_score_calibrated-metric_score:.4f}')
```

### Best Practices

- Use `calibrate_decision_threshold=True` during fitting to automatically calibrate
- The default `calibrate_decision_threshold="auto"` applies calibration when beneficial
- Custom thresholds can be used during prediction:
  ```python
  y_pred = predictor.predict(test_data)  # Uses predictor.decision_threshold
  y_pred_08 = predictor.predict(test_data, decision_threshold=0.8)  # Uses custom threshold
  y_pred_proba = predictor.predict_proba(test_data)
  y_pred = predictor.predict_from_proba(y_pred_proba)  # Same as .predict()
  ```

# Prediction Options (Inference)

## Loading a Trained Predictor

```python
predictor = TabularPredictor.load(save_path)  # Load previously trained predictor
```

You can train models on one machine and deploy on another by copying the `save_path` folder.

## Making Predictions

Check required feature columns:
```python
predictor.features()  # Returns list of feature columns needed for prediction
```

Predict on a single example:
```python
datapoint = test_data_nolabel.iloc[[0]]  # Use [[0]] not [0] to get DataFrame not Series
print(datapoint)
predictor.predict(datapoint)
```

Get predicted probabilities:
```python
predictor.predict_proba(datapoint)  # Returns DataFrame with class probabilities
```

## Model Selection and Evaluation

View the best model:
```python
predictor.model_best  # Shows which model AutoGluon considers most accurate
```

Evaluate all trained models:
```python
predictor.leaderboard(test_data)  # Basic leaderboard
predictor.leaderboard(extra_info=True)  # Detailed model information
predictor.leaderboard(test_data, extra_metrics=['accuracy', 'balanced_accuracy', 'log_loss'])
```

**Important note**: Metrics like `log_loss` are shown in `higher_is_better` form, so values will be negative. Also, `log_loss` can be `-inf` when models weren't optimized for it.

Use a specific model for prediction:
```python
model_to_use = predictor.model_names()[0]  # Select first model
model_pred = predictor.predict(datapoint, model=model_to_use)
```

Access model information:
```python
specific_model = predictor._trainer.load_model(model_to_use)
model_info = specific_model.get_info()
predictor_information = predictor.info()
```

## Evaluating Predictions

```python
# Evaluate predictions against ground truth
y_pred_proba = predictor.predict_proba(test_data_nolabel)
perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred_proba)

# Shorthand if label column is in test_data
perf = predictor.evaluate(test_data)
```

## Interpretability (Feature Importance)

```python
predictor.feature_importance(test_data)
```

Feature importance is computed via permutation-shuffling, quantifying performance drop when a column's values are randomly shuffled. Features with non-positive importance scores may be candidates for removal.

For local explanations of specific predictions, Shapley values can be used (see example notebooks).

# Accelerating Inference in AutoGluon

## Inference Optimization Options

| Optimization | Speedup | Cost | Notes |
|:-------------|:--------|:-----|:------|
| refit_full | 8x-160x | -Quality, +FitTime | Only with bagging enabled |
| persist | Up to 10x | ++MemoryUsage | Best for online inference |
| infer_limit | Up to 50x | -Quality | Use with refit_full if bagging enabled |
| distill | ~Equal to refit_full + infer_limit | --Quality, ++FitTime | Not compatible with other methods |
| feature pruning | Up to 1.5x | -Quality?, ++FitTime | Depends on feature importance |
| faster hardware | Up to 3x | +Hardware | EC2 c6i.2xlarge ~1.6x faster than m5.2xlarge |

## Optimization Priority

**With bagging enabled:**
1. refit_full
2. persist
3. infer_limit

**Without bagging:**
1. persist
2. infer_limit

## Keeping Models in Memory

```python
# Load all models into memory for faster repeated predictions
predictor.persist()

# Make predictions on individual datapoints
num_test = 20
preds = np.array(['']*num_test, dtype='object')
for i in range(num_test):
    datapoint = test_data_nolabel.iloc[[i]]
    pred_numpy = predictor.predict(datapoint, as_pandas=False)
    preds[i] = pred_numpy[0]

# Free memory when done
predictor.unpersist()
```

## Setting Inference Speed Constraints During Training

```python
# Set inference speed constraint: 0.05 ms per row (20,000 rows/second)
infer_limit = 0.00005

# Batch inference mode (easier to satisfy constraint)
infer_limit_batch_size = 10000

# For online inference, use infer_limit_batch_size = 1

...(truncated)