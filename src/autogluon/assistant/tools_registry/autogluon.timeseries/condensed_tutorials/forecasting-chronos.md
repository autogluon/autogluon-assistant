# Condensed: ```python

Summary: This tutorial covers implementing AutoGluon's Chronos-2 foundation models for time series forecasting tasks. It provides code for zero-shot forecasting (univariate/multivariate), covariate integration (past-only and known-future features), fine-tuning with LoRA adapters, backtesting, and model evaluation. Key techniques include using TimeSeriesPredictor with presets ("chronos2", "chronos2_ensemble"), configuring hyperparameters for fine-tuning (learning rate, steps, batch size, mode), handling covariates through `known_covariates_names` parameter, generating probabilistic forecasts with custom quantiles, and computing feature importance. Helps with building forecasting pipelines, comparing zero-shot vs fine-tuned models, visualizing predictions, and optimizing model performance through cross-learning mode and ensemble configurations.

*This is a condensed version that preserves essential implementation details and context.*

# Chronos-2 Implementation Guide

## Installation

```python
!pip install uv
!uv pip install -q autogluon.timeseries --system
!uv pip uninstall -q torchaudio torchvision torchtext --system
```

## Key Concepts

**Zero-shot inference**: Chronos models don't `fit` data by default. All computation happens during `predict()`, similar to ETS/ARIMA.

### Model Comparison

| Capability | Chronos | Chronos-Bolt | Chronos-2 |
|------------|---------|--------------|-----------|
| Univariate Forecasting | ✅ | ✅ | ✅ |
| Cross-learning across items | ❌ | ❌ | ✅ |
| Multivariate Forecasting | ❌ | ❌ | ✅ |
| Past-only covariates | ❌ | ❌ | ✅ |
| Known future covariates | 🧩 | 🧩 | ✅ |
| Fine-tuning support | ✅ | ✅ | ✅ |
| Max Context Length | 512 | 2048 | 8192 |
| Max Prediction Length | 64 | 64 | 1024 |

**Presets**: Use `"chronos2_small"` or `"chronos2"` (recommended), or `"bolt_tiny"`, `"bolt_mini"`, `"bolt_small"`, `"bolt_base"`.

## Zero-shot Forecasting

### Basic Usage

```python
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

# Load data
data = TimeSeriesDataFrame.from_path("path/to/data.csv")

# Split data
prediction_length = 48
train_data, test_data = data.train_test_split(prediction_length)

# Create predictor (no actual fitting occurs)
predictor = TimeSeriesPredictor(prediction_length=prediction_length).fit(
    train_data,
    presets="chronos2",
)

# Generate predictions
predictions = predictor.predict(train_data)
```

**Custom quantiles**:
```python
TimeSeriesPredictor(..., quantile_levels=[0.05, 0.1, 0.5, 0.9, 0.95])
```

### Backtesting & Visualization

```python
# Multiple windows
predictions_per_window = predictor.backtest_predictions(
    test_data, 
    num_val_windows=num_test_windows
)

# Plot
predictor.plot(test_data, predictions, max_history_length=300, item_ids=item_ids)
```

## Forecasting with Covariates

Chronos-2 supports **past-only** and **known-future** covariates (real/categorical).

```python
predictor = TimeSeriesPredictor(
    prediction_length=24,
    target="load",
    known_covariates_names=["airtemperature", "dewtemperature", "sealvlpressure"],
    eval_metric="MASE",
).fit(
    train_data,
    hyperparameters={"Chronos": {}, "Chronos2": {}},
    enable_ensemble=False,
)

# Evaluate
predictor.leaderboard(test_data)

# Feature importance
predictor.feature_importance(test_data, model="Chronos2", relative_scores=True)
```

**Note**: Other columns automatically used as past-only covariates.

## Fine-tuning

### Basic Fine-tuning

⚠️ **Multi-GPU Warning**: Set `CUDA_VISIBLE_DEVICES` to use only one GPU.

```python
predictor = TimeSeriesPredictor(
    prediction_length=prediction_length,
    target="load",
    known_covariates_names=["airtemperature", "dewtemperature", "sealvlpressure"],
    eval_metric="MASE",
).fit(
    train_data=train_data,
    hyperparameters={
        "Chronos2": [
            {"ag_args": {"name_suffix": "ZeroShot"}},  # Zero-shot
            {"fine_tune": True, "ag_args": {"name_suffix": "FineTuned"}},  # Fine-tuned
        ]
    },
    time_limit=300,
    enable_ensemble=False,
)
```

**Default**: Uses LoRA (low-rank adapter) for reduced memory/disk usage.

### Advanced Fine-tuning Parameters

```python
hyperparameters={
    "Chronos2": {
        "fine_tune": True,
        "fine_tune_mode": "full",  # or "lora" (default)
        "fine_tune_lr": 1e-4,
        "fine_tune_steps": 2000,
        "fine_tune_batch_size": 32
    }
}
```

### Ensemble Preset

```python
predictor = TimeSeriesPredictor(prediction_length=prediction_length).fit(
    train_data,
    presets="chronos2_ensemble",  # Zero-shot + fine-tuned ensemble
)
```

## Best Practices

**Fine-tuning recommendations**:
- Use when you have >100 time series with median history > `3 * prediction_length`
- Limited data can cause overfitting
- Experiment with hyperparameters if accuracy degrades
- Consider `chronos2_ensemble` preset for robustness

**Hardware**: GPU recommended (tested on AWS `g5.2xlarge` with A10G, 24GB GPU memory). CPU/consumer GPUs work for smaller models.

**Cross-learning mode**: Enabled by default, makes predictions sensitive to `batch_size`. Disable with:
```python
hyperparameters={"Chronos2": {"cross_learning": False}}
```

## Key Configuration Parameters

- `prediction_length`: Forecast horizon
- `target`: Column to forecast
- `known_covariates_names`: Future-known features
- `eval_metric`: Evaluation metric (e.g., "MASE")
- `quantile_levels`: Custom quantile levels
- `time_limit`: Training time limit (seconds)
- `enable_ensemble`: Enable/disable ensembling