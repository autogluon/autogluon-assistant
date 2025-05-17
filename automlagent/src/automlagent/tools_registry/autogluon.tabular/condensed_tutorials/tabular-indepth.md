# Condensed: AutoGluon Tabular - In Depth

Summary: This tutorial provides comprehensive implementation guidance for AutoGluon's tabular machine learning capabilities, covering model training, optimization, and deployment. It demonstrates techniques for hyperparameter configuration, model ensembling, decision threshold calibration, inference acceleration, and memory optimization. Key functionalities include automated model stacking/bagging, feature importance analysis, model persistence, and various optimization strategies (refit_full, persist, infer_limit). The tutorial helps with tasks like efficient model training, prediction acceleration (up to 160x speedup), memory usage reduction, and deployment optimization. It's particularly useful for implementing production-ready AutoML solutions that balance accuracy, inference speed, and resource constraints.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on key implementation details and concepts:

# AutoGluon Tabular - Core Implementation Details

## Key Setup
```python
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.common import space

# Load data
train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')
```

## Hyperparameter Configuration

### Important Notes
- Hyperparameter tuning usually unnecessary; `presets="best_quality"` typically works best
- Custom validation data only needed if test distribution differs from training

### Core Configuration Example
```python
# Neural Network hyperparameters
nn_options = {
    'num_epochs': 10,
    'learning_rate': space.Real(1e-4, 1e-2, default=5e-4, log=True),
    'activation': space.Categorical('relu', 'softrelu', 'tanh'),
    'dropout_prob': space.Real(0.0, 0.5, default=0.1),
}

# LightGBM hyperparameters
gbm_options = {
    'num_boost_round': 100,
    'num_leaves': space.Int(lower=26, upper=66, default=36),
}

# Combined hyperparameter configuration
hyperparameters = {
    'GBM': gbm_options,
    'NN_TORCH': nn_options,
}
```

### HPO Settings
```python
hyperparameter_tune_kwargs = {
    'num_trials': 5,  # max configurations to try
    'scheduler': 'local',
    'searcher': 'auto',
}
```

## Model Training
```python
predictor = TabularPredictor(label=label, eval_metric='accuracy').fit(
    train_data,
    time_limit=2*60,  # 2 minutes
    hyperparameters=hyperparameters,
    hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
)
```

## Best Practices
1. Start with default arguments in `TabularPredictor()` and `fit()`
2. Then experiment with:
   - `eval_metric`
   - `presets`
   - `hyperparameter_tune_kwargs`
   - `num_stack_levels`
   - `num_bag_folds`

3. For better performance:
   - Increase `subsample_size`
   - Increase `num_epochs` and `num_boost_round`
   - Extend `time_limit`
   - Use `verbosity=3` for detailed output

## Prediction
```python
y_pred = predictor.predict(test_data_nolabel)
perf = predictor.evaluate(test_data, auxiliary_metrics=False)
results = predictor.fit_summary()  # View training details
```

Here's the condensed tutorial focusing on key implementation details and practices:

# Model Ensembling and Decision Threshold Calibration

## Stacking and Bagging
- Use `num_bag_folds=5-10` and `num_stack_levels=1` to improve performance
- Key considerations:
  - Don't provide `tuning_data` with stacking/bagging
  - Use `auto_stack=True` for automatic optimization
  - `num_bag_sets` controls bagging repetition

```python
# Basic stacking/bagging implementation
predictor = TabularPredictor(label=label, eval_metric=metric).fit(
    train_data,
    num_bag_folds=5,
    num_bag_sets=1,
    num_stack_levels=1
)

# Auto-stacking implementation
predictor = TabularPredictor(label=label, eval_metric='balanced_accuracy').fit(
    train_data,
    auto_stack=True
)
```

## Decision Threshold Calibration

### Key Features:
- Improves metrics like `f1` and `balanced_accuracy` 
- Can be applied during or after model fitting
- Different thresholds optimize different metrics

### Implementation Options:

1. **Post-fit calibration**:
```python
# Calibrate and set threshold
calibrated_threshold = predictor.calibrate_decision_threshold()
predictor.set_decision_threshold(calibrated_threshold)

# Calibrate for specific metric
threshold = predictor.calibrate_decision_threshold(metric='f1')
```

2. **During fit**:
```python
predictor.fit(
    train_data,
    calibrate_decision_threshold=True  # or "auto" (default)
)
```

### Prediction Methods:
```python
# Standard prediction
y_pred = predictor.predict(test_data)

# Custom threshold prediction
y_pred_custom = predictor.predict(test_data, decision_threshold=0.8)

# Two-step prediction
y_pred_proba = predictor.predict_proba(test_data)
y_pred = predictor.predict_from_proba(y_pred_proba)
```

### Best Practices:
- Keep default `calibrate_decision_threshold="auto"`
- Be aware of metric trade-offs when calibrating
- Consider using `auto_stack=True` for optimal performance
- Stacking/bagging often outperforms hyperparameter-tuning alone

Here's the condensed version focusing on key implementation details and practices:

# Prediction and Model Management

## Loading Saved Models
```python
predictor = TabularPredictor.load(save_path)
```
- Models can be deployed by copying the `save_path` folder to new machines
- Use `predictor.features()` to see required feature columns

## Making Predictions
```python
# Single prediction
datapoint = test_data_nolabel.iloc[[0]]  # Use [[]] for DataFrame
predictor.predict(datapoint)

# Probability predictions
predictor.predict_proba(datapoint)
```

## Model Evaluation and Selection
```python
# View all models' performance
predictor.leaderboard(test_data)

# Detailed model information
predictor.leaderboard(extra_info=True)

# Multiple metrics evaluation
predictor.leaderboard(test_data, extra_metrics=['accuracy', 'balanced_accuracy', 'log_loss'])
```

**Important Notes:**
- Metrics are always shown in `higher_is_better` form (negative for log_loss, RMSE)
- `log_loss` can be `-inf` if models weren't optimized for it
- Avoid using `log_loss` as a secondary metric

## Using Specific Models
```python
model_to_use = predictor.model_names()[i]
model_pred = predictor.predict(datapoint, model=model_to_use)
```

## Model Evaluation
```python
# Evaluate predictions
y_pred_proba = predictor.predict_proba(test_data_nolabel)
perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred_proba)

# Shorthand evaluation
perf = predictor.evaluate(test_data)
```

## Feature Importance
```python
predictor.feature_importance(test_data)
```

**Key Points:**
- Uses permutation-shuffling method
- Negative scores indicate potentially harmful features
- For local explanations, use Shapley values (see example notebooks)
- Features with non-positive importance scores might be worth removing

**Best Practices:**
1. Keep save_path for model portability
2. Use DataFrame format for single predictions
3. Consider model-specific tradeoffs (accuracy vs. inference speed)
4. Be cautious with log_loss as a metric
5. Use feature importance to identify and remove harmful features

Here's the condensed version of the inference acceleration techniques in AutoGluon:

# Accelerating Inference in AutoGluon

## Key Optimization Methods (In Priority Order)

### With Bagging Enabled:
1. refit_full (8x-160x speedup)
2. persist (up to 10x speedup)
3. infer_limit (up to 50x speedup)

### Without Bagging:
1. persist
2. infer_limit

## Implementation Details

### 1. Model Persistence
```python
# Load models into memory
predictor.persist()

# Make predictions
for i in range(num_test):
    datapoint = test_data_nolabel.iloc[[i]]
    pred_numpy = predictor.predict(datapoint, as_pandas=False)

# Free memory
predictor.unpersist()
```

### 2. Inference Speed Constraints
```python
# Configure inference limits
predictor_infer_limit = TabularPredictor(label=label, eval_metric=metric).fit(
    train_data=train_data,
    time_limit=30,
    infer_limit=0.00005,  # 0.05 ms per row
    infer_limit_batch_size=10000,  # batch size for inference
)
```

## Critical Parameters
- `infer_limit`: Time in seconds to predict 1 row
- `infer_limit_batch_size`: Batch size for inference calculations
  - Use 10000 for batch inference
  - Use 1 for online inference (harder to optimize)

## Best Practices
1. Always use `refit_full` if bagging is enabled
2. Persist models for repeated predictions
3. Set appropriate batch sizes based on use case
4. Consider hardware optimization before complex manual tuning

## Important Notes
- Online inference (batch_size=1) is significantly slower than batch inference
- Memory usage increases with model persistence
- Quality-speed tradeoffs exist for most optimization methods
- Manual preprocessing and hyperparameter tuning should be last resort options

This condensed version maintains all critical implementation details while focusing on practical application and best practices.

Here's the condensed version focusing on key implementation details and best practices:

# Model Optimization and Inference Speed

## Testing Inference Speed
```python
# Test inference speed against constraints
test_data_batch = test_data.sample(infer_limit_batch_size, replace=True, ignore_index=True)

time_start = time.time()
predictor_infer_limit.predict(test_data_batch)
time_end = time.time()

infer_time_per_row = (time_end - time_start) / len(test_data_batch)
rows_per_second = 1 / infer_time_per_row
```

## Optimization Techniques

### 1. Creating Smaller Ensembles
```python
# Generate alternative ensembles with different speed-accuracy tradeoffs
additional_ensembles = predictor.fit_weighted_ensemble(expand_pareto_frontier=True)

# Use specific model for prediction
model_for_prediction = additional_ensembles[0]
predictions = predictor.predict(test_data, model=model_for_prediction)
```

### 2. Collapsing Bagged Ensembles
```python
# Collapse multiple bagged models into single model
refit_model_map = predictor.refit_full()
```
**Key Benefit**: Reduces memory/latency requirements but may impact accuracy

### 3. Model Distillation
```python
# Train smaller model to mimic ensemble predictions
student_models = predictor.distill(time_limit=30)
preds_student = predictor.predict(test_data_nolabel, model=student_models[0])
```

### 4. Lightweight Configuration Options
```python
# Use lightweight presets
predictor_light = TabularPredictor(label=label, eval_metric=metric).fit(
    train_data, 
    presets=['good_quality', 'optimize_for_deployment']
)

# Use lightweight hyperparameters
predictor_light = TabularPredictor(label=label, eval_metric=metric).fit(
    train_data, 
    hyperparameters='very_light'
)


...(truncated)