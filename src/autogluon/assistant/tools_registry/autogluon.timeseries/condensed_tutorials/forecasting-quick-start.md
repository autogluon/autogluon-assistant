# Condensed: AutoGluon Time Series - Forecasting Quick Start

Summary: This tutorial demonstrates implementing time series forecasting using AutoGluon's TimeSeriesPredictor framework. It covers essential techniques for loading time series data in long format, configuring and training forecasting models with different quality presets, and generating probabilistic predictions. The tutorial helps with tasks like converting data to TimeSeriesDataFrame format, setting up model training with various presets (fast_training to best_quality), and evaluating model performance. Key features include handling multiple time series, configurable prediction horizons, support for various models (from simple baselines to deep learning), probabilistic forecasting with quantiles, and model evaluation through leaderboards. The implementation knowledge spans data formatting requirements, model configuration options, and best practices for forecast generation.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details:

# AutoGluon Time Series - Forecasting Quick Start

## Key Components
- `TimeSeriesDataFrame`: Stores multiple time series datasets
- `TimeSeriesPredictor`: Handles model fitting, tuning, and forecasting

## Implementation Details

### 1. Setup and Data Loading
```python
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

# Load data and convert to TimeSeriesDataFrame
train_data = TimeSeriesDataFrame.from_data_frame(
    df,
    id_column="item_id",
    timestamp_column="timestamp"
)
```

**Required Data Format:**
- Long format with columns for:
  - Unique ID (`item_id`)
  - Timestamp (`timestamp`)
  - Target value (`target`)

### 2. Model Training
```python
predictor = TimeSeriesPredictor(
    prediction_length=48,  # Forecast horizon
    path="autogluon-m4-hourly",
    target="target",
    eval_metric="MASE"
)

predictor.fit(
    train_data,
    presets="medium_quality",
    time_limit=600
)
```

**Important Configurations:**
- `prediction_length`: Number of future timesteps to forecast
- `presets`: Available options:
  - `"fast_training"`
  - `"medium_quality"` (includes baselines, statistical models, tree-based models, deep learning)
  - `"high_quality"`
  - `"best_quality"`

### 3. Generating Forecasts
```python
predictions = predictor.predict(train_data)
```

**Output Features:**
- Produces probabilistic forecasts
- Includes mean predictions and quantiles
- Forecasts `prediction_length` timesteps ahead

### 4. Model Evaluation
```python
predictor.leaderboard(test_data)
```

## Best Practices
1. Ensure data is in correct long format
2. Choose appropriate prediction length based on data frequency
3. Select presets based on accuracy vs. training time requirements
4. Use quantile forecasts to understand prediction uncertainty

## Supported Models
- Simple baselines (Naive, SeasonalNaive)
- Statistical models (ETS, Theta)
- Tree-based models (RecursiveTabular, DirectTabular)
- Deep learning (TemporalFusionTransformer)
- Weighted ensemble combinations

## Important Notes
- AutoGluon generates individual forecasts for each time series without modeling inter-series interactions
- Higher quality presets typically produce better forecasts but require longer training times
- Models are ranked based on performance on internal validation set
- Leaderboard scores are multiplied by -1 (higher scores = better performance)