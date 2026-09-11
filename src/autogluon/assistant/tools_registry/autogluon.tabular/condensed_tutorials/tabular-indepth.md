# Condensed: ```python

Summary: This tutorial covers advanced AutoGluon Tabular techniques for production ML systems, including model ensembling (bagging with `num_bag_folds`, stacking with `num_stack_levels`), binary classification decision threshold calibration for metric optimization, comprehensive model evaluation via leaderboards with custom metrics, and feature importance analysis. It provides detailed inference acceleration strategies prioritized as: `refit_full()` for collapsing bagged ensembles (8x speedup), `persist()` for keeping models in memory (10x speedup in online inference), and `infer_limit` constraints during training (50x+ speedup). Additional optimization techniques include Pareto-optimal weighted ensembles, model distillation, preprocessing caching, and memory/disk management strategies. The tutorial helps with tasks like deploying fast prediction systems, optimizing model accuracy-latency tradeoffs, reducing resource usage, cross-machine model deployment, and implementing interpretable ML solutions with proper metric calibration.

*This is a condensed version that preserves essential implementation details and context.*

# AutoGluon Tabular: Advanced Techniques

## Installation & Setup

```python
!pip install autogluon.tabular[all]

from autogluon.tabular import TabularDataset, TabularPredictor

train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
train_data = train_data.sample(n=1000, random_state=0)  # Subsample for faster demo
test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')

label = 'class'
metric = "balanced_accuracy"
```

## Model Ensembling: Stacking & Bagging

**Key Technique**: [Bagging and stack-ensembling](https://arxiv.org/abs/2003.06505) boost predictive performance but increase training time and resource usage.

### Implementation

```python
predictor = TabularPredictor(label=label, eval_metric=metric, path='agModels-predictClass').fit(
    train_data,
    calibrate_decision_threshold=False,  # Disabled for demonstration
    num_bag_folds=8,        # k-fold bagging
    num_bag_sets=1,         # Repeat bagging to reduce variance (increases resources)
    num_stack_levels=1,     # Stack ensemble layers
    time_limit=180
)
# Alternative: auto_stack=True or presets="best_quality"
```

**⚠️ Best Practices**:
- **Don't provide `tuning_data`** when stacking/bagging—provide all data as `train_data`
- Use `auto_stack=True` for automatic optimization instead of manual tuning
- Increasing `num_bag_sets` improves accuracy but substantially increases training time, latency, and memory/disk usage

## Decision Threshold Calibration

**Critical Concept**: For binary classification, adjusting the decision threshold from default 0.5 can significantly improve metrics like `"f1"` and `"balanced_accuracy"`.

### Basic Calibration

```python
# Before calibration (default threshold = 0.5)
scores = predictor.evaluate(test_data)

# Calibrate and apply
calibrated_decision_threshold = predictor.calibrate_decision_threshold()
predictor.set_decision_threshold(calibrated_decision_threshold)

scores_calibrated = predictor.evaluate(test_data)
```

### Metric-Specific Calibration

```python
predictor.set_decision_threshold(0.5)  # Reset to default

for metric_name in ['f1', 'balanced_accuracy', 'mcc']:
    calibrated_threshold = predictor.calibrate_decision_threshold(
        metric=metric_name, verbose=False
    )
    metric_score_calibrated = predictor.evaluate(
        test_data, decision_threshold=calibrated_threshold, silent=True
    )[metric_name]
```

**⚠️ Important Tradeoff**: Calibrating for one metric often improves that metric but may harm others (e.g., optimizing "balanced_accuracy" may reduce "accuracy").

### Auto-Calibration During Fit

```python
# Automatic calibration during training
predictor.fit(..., calibrate_decision_threshold=True)

# Default behavior (recommended)
predictor.fit(..., calibrate_decision_threshold="auto")  # Applied when beneficial
```

### Prediction with Custom Thresholds

```python
# Use predictor's stored threshold
y_pred = predictor.predict(test_data)

# Override with specific threshold
y_pred_08 = predictor.predict(test_data, decision_threshold=0.8)

# From probability predictions
y_pred_proba = predictor.predict_proba(test_data)
y_pred = predictor.predict_from_proba(y_pred_proba)  # Uses stored threshold
y_pred_08 = predictor.predict_from_proba(y_pred_proba, decision_threshold=0.8)
```

**✅ Recommendation**: Keep default `calibrate_decision_threshold="auto"` for most use cases.

# Prediction & Interpretability

## Loading Trained Models

```python
# Load previously trained predictor from disk
predictor = TabularPredictor.load(save_path)  # save_path from fit() call

# Get required feature columns
predictor.features()
```

**✅ Deployment**: Copy `save_path` folder to new machine and load with new path for cross-machine deployment.

## Making Predictions

### Single Example Prediction

```python
# Must use DataFrame (not Series)
datapoint = X_test.iloc[[0]]  # Note: .iloc[0] returns Series, won't work

# Predict class
predictor.predict(datapoint)

# Predict probabilities
predictor.predict_proba(datapoint)  # Returns DataFrame with class probabilities
```

### Model Selection

```python
# Default: uses most accurate model (usually ensemble)
predictor.model_best

# Use specific model (e.g., for lower latency)
model_to_use = predictor.model_names()[0]
model_pred = predictor.predict(datapoint, model=model_to_use)
```

**Note**: An AutoGluon 'model' can be a single model, bagged ensemble, weighted ensemble, or stacker model.

## Model Evaluation & Leaderboard

### Basic Leaderboard

```python
# Evaluate all models on test data
predictor.leaderboard(test_data)
# Shows: score_test, score_val, pred_time_test, pred_time_val, fit_time

# Use validation data only (no new data needed)
predictor.leaderboard()
```

### Extended Leaderboard

```python
# Show additional model properties
predictor.leaderboard(extra_info=True)
# Shows: num_features, ancestors, memory_size_w_ancestors, etc.

# Add custom metrics
predictor.leaderboard(test_data, extra_metrics=['accuracy', 'balanced_accuracy', 'log_loss'])
```

**⚠️ Important**: Metrics like `log_loss` and `root_mean_squared_error` are **sign-flipped** (negative values) because AutoGluon displays all metrics in `higher_is_better` form.

## Model Information Access

```python
# Predictor-level information
predictor_information = predictor.info()

# Model-specific information
model_info = predictor.model_info(model_to_use)

# Direct model access
model_info_alternative = predictor._trainer.load_model(model_to_use).get_info()
```

## Evaluation Methods

```python
# Evaluate predictions with ground truth
y_pred_proba = predictor.predict_proba(X_test)
predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred_proba)

# Shorthand (if label column in DataFrame)
predictor.evaluate(test_data)
```

## Feature Importance

```python
# Global feature importance via permutation-shuffling
predictor.feature_importance(test_data)
```

**Key Insights**:
- **Quantifies**: Drop in performance when feature values are randomly shuffled
- **Top features**: Contribute most to accuracy
- **Non-positive scores**: Features that don't contribute or may harm performance (consider removing)
- **Scope**: Global behavior across all predictions

### Local Explanations

For individual prediction explanations, use [Shapley values](https://github.com/slundberg/shap/). See [example notebooks](https://github.com/autogluon/autogluon/tree/master/examples/tabular/interpret).

**📚 Recommended Reading**: [The Mythos of Model Interpretability](https://dl.acm.org/doi/pdf/10.1145/3236386.3241340) by Zachary Lipton on interpretability claims.

# Accelerating Inference

## Optimization Priority Table

| Optimization | Speedup | Cost | Notes |
|:------------|:--------|:-----|:------|
| **refit_full** | 8x (with bagging) | -Quality | Only effective with bagging enabled |
| **persist** | Up to 10x (online-inference) | +Memory | Most effective for online-inference, not batch |
| **infer_limit** | Up to 50x+ | -Quality | Best combined with refit_full |
| **feature pruning** | ~1.5x max | -Quality? | Use `predictor.feature_importance(test_data)` to identify removable features |
| **faster hardware** | ~3x max | +Hardware | e.g., EC2 c6i.2xlarge is ~1.6x faster than m5.2xlarge |
| **manual hyperparameters** | ~2x max | Manual effort | Reduce trees in LightGBM/XGBoost/CatBoost (not recommended) |
| **manual preprocessing** | ~1.2x max | Manual effort | Online-inference only (not recommended) |

**✅ Recommended Order**: 
1. refit_full
2. persist  
3. infer_limit

## persist: Keep Models in Memory

**Default behavior**: Models load one at a time on-demand (slower but robust).

### Implementation

```python
import time
import numpy as np

# Benchmark without persist
num_test = 20
time_start = time.time()
for i in range(num_test):
    datapoint = X_test.iloc[[i]]
    pred_numpy = predictor.predict(datapoint, as_pandas=False)
time_without_persist = (time.time() - time_start) / num_test

# Benchmark with persist
predictor.persist()  # Load all required models into memory

time_start = time.time()
for i in range(num_test):
    datapoint = X_test.iloc[[i]]
    pred_numpy = predictor.predict(datapoint, as_pandas=False)
time_with_persist = (time.time() - time_start) / num_test

predictor.unpersist()  # Free memory, future predictions load from disk
```

### Persist Options

```python
predictor.persist()              # Persist models needed for best predictor
predictor.persist(models='all')  # Persist ALL trained models
predictor.persist('best')        # Persist only minimum required for prediction (optimized)
```

**⚠️ Note**: Speedup requires sufficient memory. Most effective for online-inference (repeated single predictions).

## infer_limit: Inference Speed Constraint

Specify latency constraints **before fitting** to train models that satisfy speed requirements.

### Key Parameters

- **`infer_limit`**: Time (seconds) to predict 1 row
  - Example: `0.05` = 50ms/row = 20 rows/second throughput
- **`infer_limit_batch_size`**: Rows passed at once during prediction
  - `1` = online-inference (harder to satisfy)
  - `10000` = batch-inference (easier to satisfy)

### Implementation

```python
infer_limit = 0.00005  # 0.05ms per row (20,000 rows/second)
infer_limit_batch_size = 10000  # Batch inference

predictor_infer_limit = TabularPredictor(label=label, eval_metric=metric).fit(
    train_data=train_data,
    time_limit=30,
    infer_limit=infer_limit,
    infer_limit_batch_size=infer_limit_batch_size,
)

# IMPORTANT: Call refit_full if bagging was enabled
# predictor_infer_limit.refit_full()

# IMPORTANT: Persist models to match fit-time inference speed calculations
predictor_infer_limit.persist()
# Or optimized: predictor_infer_limit.persist('best')
```

**⚠️ Critical Notes**:
- `infer_limit` assumes `refit_full()` will be called after `fit()`
- Models must be persisted to align with fit-time speed calculations
- `infer_limit < 0.02` with `infer_limit_batch_size=1` is difficult to satisfy

### Validate Constraint Satisfaction

```python
test_data_batch = test_data.sample(infer_limit_batch_size, replace=True, ignore_index=True)

time_start = time.time()
predictor_infer_limit.predict(test_data_batch)
time_end = time.time()

infer_time_per_row = (time_end - time_start) / len(test_data_batch)
rows_per_second = 1 / infer_time_per_row
infer_time_per_row_ratio = infer_time_per_row / infer_limit
is_constraint_satisfied = infer_time_per_row_ratio <= 1

print(f'Throughput: {round(rows_per_second, 1)} rows/sec (Target: {1/infer_limit})')
print(f'Uses {round(infer_time_per_row_ratio * 100, 1)}% of infer_limit')
print(f'Constraint satisfied: {is_constraint_satisfied}')
```

# Advanced Inference Optimization

## Smaller Ensembles via Weighted Ensembles

Create alternative ensembles with different accuracy-speed tradeoffs **without retraining**:

```python
# Generate Pareto-optimal ensembles
additional_ensembles = predictor.fit_weighted_ensemble(expand_pareto_frontier=True)

# View most accurate model for each inference latency
predictor.leaderboard(only_pareto_frontier=True)

# Use specific ensemble for prediction
model_for_prediction = additional_ensembles[0]
predictions = predictor.predict(test_data, model=model_for_prediction)

# Cleanup
predictor.delete_models(models_to_delete=additional_ensembles)
```

**Key Concept**: Pareto frontier shows optimal accuracy-latency tradeoffs.

## refit_full: Collapse Bagged Ensembles

Collapse ~10 bagged model copies into a single model trained on full dataset:

```python
refit_model_map = predictor.refit_full()  # Refit all models
# Or: predictor.refit_full(model='specific_model')  # Refit specific model

predictor.leaderboard(test_data)
```

**Benefits**:
- Greatly reduces memory/latency (8x speedup with bagging)
- Works with non-bagged models (may improve accuracy)

**⚠️ Note**: Refit-full models have no `score_val` (trained on all data, no validation split).

## Model Distillation

Train a single "student" model to mimic ensemble "teacher" predictions:

```python
student_models = predictor.distill(time_limit=120)
predictor.leaderboard(test_data)
```

**⚠️ Warning**: May cause significant accuracy drop. Prioritize `refit_full`, `infer_limit`, and `persist` first.

## Lightweight Training Options

### Faster Presets

```python
presets = ['good_quality', 'optimize_for_deployment']
predictor_light = TabularPredictor(label=label, eval_metric=metric).fit(
    train_data, presets=presets, time_limit=30
)
```

### Exclude Slow Models

```python
excluded_model_types = ['KNN', 'NN_TORCH']  # Exclude K-Nearest Neighbors, Neural Networks
predictor_light = TabularPredictor(label=label, eval_metric=metric).fit(
    train_data, excluded_model_types=excluded_model_types, time_limit=30
)
```

## Advanced: Cache Preprocessed Data

For **repeated predictions on same data**:

```python
# Preprocess once
test_data_preprocessed = predictor.transform_features(test_data)

# Skip preprocessing in predict
predictions = predictor.predict(test_data_preprocessed, transform_features=False)
```

**⚠️ Use Case**: Only beneficial when repeatedly predicting on identical data. Consider caching predictions instead.

## Advanced: Disable Preprocessing

```python
predictor.fit(..., feature_generator=None, feature_metadata=YOUR_CUSTOM_FEATURE_METADATA)
```

**⚠️ Warning**: Removes ALL data sanitization guardrails. Only for advanced users familiar with AutoGluon.

**Use Case**: Multiple tasks reusing exact same features—preprocess once, send to multiple predictors.

## Memory Optimization

### Reduce Training Memory

```python
# Strategy 1: Exclude memory-intensive models
excluded_model_types = ['KNN', 'XT', 'RF']

# Strategy 2: Use different presets
presets = ['medium_quality', 'optimize_for_deployment']

# Strategy 3: Reduce text N-gram features
from sklearn.feature_extraction.text import CountVectorizer
from autogluon.features.generators import AutoMLPipelineFeatureGenerator

MAX_NGRAM = 1000  # Try values < 10000
feature_generator = AutoMLPipelineFeatureGenerator(
    vectorizer=CountVectorizer(min_df=30, ngram_range=(1, 3), 
                               max_features=MAX_NGRAM, dtype=np.uint8)
)
predictor = TabularPredictor(...).fit(..., feature_generator=feature_generator)

# Strategy 4: Ignore text features entirely
presets = ['medium_quality', 'ignore_text']
```

### Reduce Inference Memory

```python
# Break large test data into chunks (see FAQ)

# Unpersist models if speed not critical
predictor.unpersist()

# With bagging: use refit_full model
predictor.refit_full()
# Persist only the refit model for prediction
```

**✅ Tip**: These strategies also reduce training time.

# Disk Space Optimization

## Strategies to Reduce Disk Usage

### Clean Up Old Models

```python
# Delete previous predictor folders manually
# Default paths: "AutogluonModels/ag-[TIMESTAMP]"
# Custom paths: predictor.path
```

**⚠️ Critical**: Delete old `predictor.path` folders from previous `fit()` runs to prevent disk space exhaustion.

### Post-Training Cleanup

```python
# Delete auxiliary files from fit()
predictor.save_space()

# Keep only best model for inference (deletes files needed for fit_summary, etc.)
predictor.delete_models(models_to_keep='best')
```

### Automatic Optimization During Training

```python
# Automatically apply save_space() and delete_models() after training
presets = ['medium_quality', 'optimize_for_deployment']
predictor = TabularPredictor(label=label).fit(train_data, presets=presets)
```

**✅ Tip**: Memory reduction strategies also reduce disk usage (may impact accuracy).

---

## References

**AutoGluon-Tabular Paper**:  
Erickson et al. [AutoGluon-Tabular: Robust and Accurate AutoML for Structured Data](https://arxiv.org/abs/2003.06505). *Arxiv*, 2020.

---

## Next Steps

For deployment optimization, see: [Predicting Columns in a Table - Deployment Optimization](advanced/tabular-deployment.ipynb)