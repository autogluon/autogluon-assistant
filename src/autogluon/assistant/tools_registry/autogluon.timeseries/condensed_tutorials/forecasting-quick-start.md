# Condensed: ```python

Summary: This tutorial covers AutoGluon's time series forecasting implementation using `TimeSeriesDataFrame` and `TimeSeriesPredictor` classes. It teaches how to format panel time series data in long format (item_id, timestamp, target columns), train automated forecasting models with configurable parameters (prediction_length, presets, time_limit, eval_metric), and generate probabilistic forecasts with mean and quantile predictions. Key techniques include data preparation from pandas DataFrames, model training with quality presets (medium/high/best_quality) that automatically fit multiple models (SeasonalNaive, ETS, Theta, LightGBM-based, Chronos-2, Toto-2, ensembles), forecast visualization, and model evaluation via leaderboards. Useful for building multi-step ahead forecasting pipelines, comparing model performance on validation/test sets, and understanding that this handles independent panel forecasting (not multivariate). Critical parameters covered: prediction_length for forecast horizon, presets for model selection, and time_limit for training duration constraints.

*This is a condensed version that preserves essential implementation details and context.*

# AutoGluon Time Series Forecasting - Condensed Tutorial

## Installation

```python
!pip install uv
!uv pip install -q autogluon.timeseries --system
!uv pip uninstall -q torchaudio torchvision torchtext --system  # fix incompatible packages on Colab
```

## Core Classes

```python
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
```

- **`TimeSeriesDataFrame`**: Stores datasets with multiple time series
- **`TimeSeriesPredictor`**: Handles fitting, tuning, model selection, and forecasting

## Data Format Requirements

AutoGluon requires data in **long format** with three essential columns:
- **Unique ID** (`item_id`): int or str identifier for each time series
- **Timestamp** (`timestamp`): pandas.Timestamp or compatible format
- **Target value** (`target`): numeric values

```python
train_data = TimeSeriesDataFrame.from_data_frame(
    df,
    id_column="item_id",
    timestamp_column="timestamp"
)
```

**Key Concept**: Each individual time series is called an _item_. This is a _panel_ of time series, **not multivariate forecasting** — forecasts are generated independently for each item.

## Training Models

```python
predictor = TimeSeriesPredictor(
    prediction_length=48,      # Forecast horizon (e.g., 48 hours)
    path="autogluon-m4-hourly",
    target="target",
    eval_metric="MASE",        # Mean Absolute Scaled Error
)

predictor.fit(
    train_data,
    presets="medium_quality",  # Options: medium_quality, high_quality, best_quality
    time_limit=600,            # Training time in seconds
)
```

**Critical Parameters**:
- **`prediction_length`**: Number of future timesteps to forecast
- **`presets`**: Defines model suite (higher quality = more accurate but slower)
  - `medium_quality`: SeasonalNaive, ETS, Theta, RecursiveTabular, DirectTabular, Chronos-2, Toto-2, weighted ensemble
- **Validation**: By default, last `prediction_length` timesteps are held out for internal validation

## Generating Forecasts

```python
predictions = predictor.predict(train_data)
```

**Output**: Probabilistic forecast with:
- **Mean**: Expected future values
- **Quantiles**: Range of possible outcomes (e.g., `0.1` quantile = 10% chance value will be below this)

**Visualization**:
```python
predictor.plot(test_data, predictions, quantile_levels=[0.1, 0.9], 
               max_history_length=200, max_num_item_ids=4)
```

## Model Evaluation

```python
predictor.leaderboard(test_data)
```

**Important**: Test data must include both:
- Historical data (all except last `prediction_length` values)
- Forecast horizon (last `prediction_length` values)

**Note**: Scores are inverted (multiplied by -1) so higher values = better performance

## Best Practices
- Choose `prediction_length` based on your forecasting needs and data frequency
- Higher quality presets improve accuracy but increase training time
- AutoGluon trains as many models as possible within `time_limit`
- Best model (by validation score) is used for predictions by default