# Condensed: ```python

Summary: This tutorial introduces Chronos and Chronos-Bolt models in AutoGluon for time series forecasting. It covers implementation of zero-shot forecasting that scales linearly with time series volume, model selection across different sizes (tiny to large), and practical code examples for prediction. Key functionalities include fine-tuning options with customizable learning rates and steps, incorporating covariates through regressors (particularly CatBoost), and hardware recommendations. The tutorial demonstrates how to load data, split into train/test sets, create predictors with various configurations, generate predictions, and evaluate model performance through leaderboards—particularly useful for implementing efficient time series forecasting with minimal training requirements.

*This is a condensed version that preserves essential implementation details and context.*

# Getting Started with Chronos in AutoGluon

## Installation

```python
!pip install uv
!uv pip install -q autogluon.timeseries --system
!uv pip uninstall -q torchaudio torchvision torchtext --system # fix incompatible package versions on Colab
```

## Overview

Chronos is a pretrained model for zero-shot forecasting that differs from other AG-TS models:
- Does not truly `fit` time series data
- Computation happens during inference (like ETS or ARIMA)
- Scales linearly with the number of time series

AutoGluon supports:
- Original Chronos models (e.g., `chronos-t5-large`)
- New Chronos-Bolt⚡ models (up to 250x faster, more accurate)

## Model Selection

**Recommended presets:**
- Chronos-Bolt: `"bolt_tiny"`, `"bolt_mini"`, `"bolt_small"`, `"bolt_base"` (CPU/GPU compatible)
- Original Chronos: `"chronos_tiny"`, `"chronos_mini"`, `"chronos_small"`, `"chronos_base"`, `"chronos_large"` (sizes `small` and above require GPU)

## Zero-shot Forecasting Example

```python
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

# Load dataset
data = TimeSeriesDataFrame.from_path(
    "https://autogluon.s3.amazonaws.com/datasets/timeseries/australian_electricity_subset/test.csv"
)

# Split data
prediction_length = 48
train_data, test_data = data.train_test_split(prediction_length)

# Create predictor with Chronos-Bolt
predictor = TimeSeriesPredictor(prediction_length=prediction_length).fit(
    train_data, presets="bolt_small",
)

# Generate and visualize predictions
predictions = predictor.predict(train_data)
predictor.plot(
    data=data,
    predictions=predictions,
    item_ids=data.item_ids[:2],
    max_history_length=200,
);
```

## Fine-tuning

Compare zero-shot and fine-tuned models:

```python
predictor = TimeSeriesPredictor(prediction_length=prediction_length).fit(
    train_data=train_data,
    hyperparameters={
        "Chronos": [
            {"model_path": "bolt_small", "ag_args": {"name_suffix": "ZeroShot"}},
            {"model_path": "bolt_small", "fine_tune": True, "ag_args": {"name_suffix": "FineTuned"}},
        ]
    },
    time_limit=60,  # time limit in seconds
    enable_ensemble=False,
)

# Evaluate models
predictor.leaderboard(test_data)
```

**Advanced fine-tuning options:**
```python
predictor.fit(
    ...,
    hyperparameters={"Chronos": {"fine_tune": True, "fine_tune_lr": 1e-4, "fine_tune_steps": 2000}},
)
```

## Incorporating Covariates

Chronos is univariate but can be combined with covariate regressors to incorporate exogenous information:

```python
# Load dataset with covariates
data = TimeSeriesDataFrame.from_path(
    "https://autogluon.s3.amazonaws.com/datasets/timeseries/grocery_sales/test.csv",
)

prediction_length = 8
train_data, test_data = data.train_test_split(prediction_length=prediction_length)

# Create predictor with covariates
predictor = TimeSeriesPredictor(
    prediction_length=prediction_length,
    target="unit_sales",
    known_covariates_names=["scaled_price", "promotion_email", "promotion_homepage"],
).fit(
    train_data,
    hyperparameters={
        "Chronos": [
            # Zero-shot model WITHOUT covariates
            {
                "model_path": "bolt_small",
                "ag_args": {"name_suffix": "ZeroShot"},
            },
            # Chronos-Bolt with CatBoost covariate regressor
            {
                "model_path": "bolt_small",
                "covariate_regressor": "CAT",
                "target_scaler": "standard",
                "ag_args": {"name_suffix": "WithRegressor"},
            },
        ],
    },
    enable_ensemble=False,
    time_limit=60,
)

# Evaluate models
predictor.leaderboard(test_data)
```

## Hardware Recommendations

- For larger models: AWS `g5.2xlarge` or `p3.2xlarge` with NVIDIA A10G/V100 GPUs (16GB+ GPU memory, 32GB+ RAM)
- Chronos-Bolt models can run on CPU but with longer runtime