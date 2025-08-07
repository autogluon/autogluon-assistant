# Condensed: ```python

Summary: This tutorial demonstrates AutoGluon's time series forecasting capabilities, teaching LLMs how to implement probabilistic time series forecasting with minimal code. It covers TimeSeriesDataFrame for data preparation in long format (requiring item_id, timestamp, and target columns), TimeSeriesPredictor for model training with various quality presets, and generating probabilistic forecasts with quantiles. Key functionalities include handling multiple time series simultaneously, automatic model selection (including statistical, tree-based, and deep learning approaches), customizable forecast horizons, and model evaluation using metrics like MASE. This knowledge helps with implementing production-ready time series forecasting systems.

*This is a condensed version that preserves essential implementation details and context.*

# AutoGluon Time Series Forecasting Tutorial

## Setup and Installation

```python
!pip install uv
!uv pip install -q autogluon.timeseries --system
!uv pip uninstall -q torchaudio torchvision torchtext --system # fix incompatible package versions on Colab

import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
```

## Key Components

- **TimeSeriesDataFrame**: Stores multiple time series datasets
- **TimeSeriesPredictor**: Handles model fitting, tuning, selection, and forecasting

## Data Preparation

AutoGluon requires time series data in **long format** with three essential columns:
- Unique ID for each time series (`item_id`)
- Timestamp of observation (`timestamp`)
- Target value (`target`)

```python
df = pd.read_csv("https://autogluon.s3.amazonaws.com/datasets/timeseries/m4_hourly_subset/train.csv")

train_data = TimeSeriesDataFrame.from_data_frame(
    df,
    id_column="item_id",
    timestamp_column="timestamp"
)
```

## Training Models

```python
predictor = TimeSeriesPredictor(
    prediction_length=48,  # Forecast horizon (48 hours)
    path="autogluon-m4-hourly",
    target="target",
    eval_metric="MASE",
)

predictor.fit(
    train_data,
    presets="medium_quality",
    time_limit=600,  # 10 minutes
)
```

**Preset Options**:
- `medium_quality`: Includes baselines (`Naive`, `SeasonalNaive`), statistical models (`ETS`, `Theta`), tree-based models (`RecursiveTabular`, `DirectTabular`), deep learning (`TemporalFusionTransformer`), and weighted ensemble
- Other options: `fast_training`, `high_quality`, `best_quality`

## Generating Forecasts

```python
predictions = predictor.predict(train_data)
```

AutoGluon produces **probabilistic forecasts** with both mean predictions and quantiles of the forecast distribution.

## Visualization

```python
import matplotlib.pyplot as plt

test_data = TimeSeriesDataFrame.from_path("https://autogluon.s3.amazonaws.com/datasets/timeseries/m4_hourly_subset/test.csv")

predictor.plot(test_data, predictions, quantile_levels=[0.1, 0.9], max_history_length=200, max_num_item_ids=4)
```

## Model Evaluation

```python
# Evaluates models on test data
predictor.leaderboard(test_data)
```

Note: MASE scores are multiplied by `-1` in the leaderboard, so higher "negative MASE" values indicate better forecasts.