# Condensed: ```python

Summary: This tutorial covers AutoGluon's TimeSeriesPredictor for time series forecasting, including: (1) adding static features and covariates (known/past) to TimeSeriesDataFrame with proper item_id mapping; (2) implementing holiday features using the holidays package with custom helper functions; (3) handling irregular time indices and missing values via convert_frequency() and fill_missing_values(); (4) train-test splitting, evaluation with multiple backtest windows using cutoff parameters, and internal validation with num_val_windows; (5) configuring local models (ETS, ARIMA), global models (DeepAR, Chronos, RecursiveTabular), and ensembles; (6) using presets (medium/high/best_quality), manual hyperparameter configuration, model exclusion, and HPO with search spaces. Key for implementing production forecasting pipelines with proper validation, covariate handling, and model selection.

*This is a condensed version that preserves essential implementation details and context.*

# AutoGluon Time Series: Static Features & Covariates

## Installation & Setup

```python
!pip install uv
!uv pip install -q autogluon.timeseries --system
!uv pip uninstall -q torchaudio torchvision torchtext --system

import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
```

## Static Features

Load data with static features (e.g., M4 Daily dataset with categorical domain feature):

```python
df = pd.read_csv("https://autogluon.s3.amazonaws.com/datasets/timeseries/m4_daily_subset/train.csv")
static_features_df = pd.read_csv("https://autogluon.s3.amazonaws.com/datasets/timeseries/m4_daily_subset/metadata.csv")
```

**Key**: Static features require `item_id` column to map features to time series.

Create `TimeSeriesDataFrame` with static features:

```python
train_data = TimeSeriesDataFrame.from_data_frame(
    df,
    id_column="item_id",
    timestamp_column="timestamp",
    static_features_df=static_features_df,
)
# Or attach later:
train_data.static_features = static_features_df
```

## Covariates

### Known Covariates (Future-Available Features)

Add features known at prediction time (e.g., weekend indicator):

```python
import numpy as np
train_data["log_target"] = np.log(train_data["target"])  # past covariate

WEEKEND_INDICES = [5, 6]
timestamps = train_data.index.get_level_values("timestamp")
train_data["weekend"] = timestamps.weekday.isin(WEEKEND_INDICES).astype(float)
```

Specify covariates in predictor:

```python
predictor = TimeSeriesPredictor(
    prediction_length=14,
    target="target",
    known_covariates_names=["weekend"],  # must be known at prediction time
).fit(train_data)
# Remaining columns auto-interpreted as past covariates (e.g., 'log_target')
```

### Prediction with Known Covariates

Generate future covariate values:

```python
known_covariates = predictor.make_future_data_frame(train_data)
known_covariates["weekend"] = known_covariates["timestamp"].dt.weekday.isin(WEEKEND_INDICES).astype(float)

predictions = predictor.predict(train_data, known_covariates=known_covariates)
```

**Requirements for `known_covariates`**:
- Must include all columns in `predictor.known_covariates_names`
- Must include all `item_id` values from `train_data`
- Must cover `prediction_length` future timestamps for each series

## Holiday Features

Install and define holidays:

```python
!pip install -q holidays
import holidays

timestamps = train_data.index.get_level_values("timestamp")
country_holidays = holidays.country_holidays(
    country="DE",  # select appropriate country
    years=range(timestamps.min().year, timestamps.max().year + 1),  # +1 for forecast horizon
)
```

Add holiday features helper function:

```python
def add_holiday_features(
    ts_df: TimeSeriesDataFrame,
    country_holidays: dict,
    include_individual_holidays: bool = True,
    include_holiday_indicator: bool = True,
) -> TimeSeriesDataFrame:
    ts_df = ts_df.copy()
    if not isinstance(ts_df, TimeSeriesDataFrame):
        ts_df = TimeSeriesDataFrame(ts_df)
    timestamps = ts_df.index.get_level_values("timestamp")
    country_holidays_df = pd.get_dummies(pd.Series(country_holidays)).astype(float)
    holidays_df = country_holidays_df.reindex(timestamps.date).fillna(0)
    if include_individual_holidays:
        ts_df[holidays_df.columns] = holidays_df.values
    if include_holiday_indicator:
        ts_df["Holiday"] = holidays_df.max(axis=1).values
    return ts_df
```

Use in training and prediction:

```python
train_data_with_holidays = add_holiday_features(train_data, country_holidays)
holiday_columns = train_data_with_holidays.columns.difference(train_data.columns)

predictor = TimeSeriesPredictor(..., known_covariates_names=holiday_columns).fit(train_data_with_holidays)

# At prediction time
known_covariates = predictor.make_future_data_frame(train_data)
known_covariates = add_holiday_features(known_covariates, country_holidays)
predictions = predictor.predict(train_data_with_holidays, known_covariates=known_covariates)
```

**Note**: See [Forecasting Model Zoo](forecasting-model-zoo.md) for models supporting static features and covariates.

# Data Format, Validation & Evaluation

## Minimum Data Length Requirements

**Default training**:
```python
predictor = TimeSeriesPredictor(prediction_length=prediction_length).fit(train_data)
```
Requires: `length >= max(prediction_length + 1, 5) + prediction_length`

**Advanced configuration**:
```python
predictor = TimeSeriesPredictor(prediction_length=prediction_length).fit(
    train_data, num_val_windows=num_val_windows, val_step_size=val_step_size
)
```
Requires: `length >= max(prediction_length + 1, 5) + prediction_length + (num_val_windows - 1) * val_step_size`

**Note**: Time series can have different lengths.

## Handling Irregular Data & Missing Values

### Irregular Time Index

Example irregular data:
```python
df_irregular = TimeSeriesDataFrame(pd.DataFrame({
    "item_id": [0, 0, 0, 1, 1],
    "timestamp": ["2022-01-01", "2022-01-02", "2022-01-04", "2022-01-01", "2022-01-04"],
    "target": [1, 2, 3, 4, 5],
}))
```

**Option 1**: Specify frequency in predictor (auto-fills gaps):
```python
predictor = TimeSeriesPredictor(..., freq="D").fit(df_irregular)
```
See [pandas frequency aliases](https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases).

**Option 2**: Manually convert frequency:
```python
df_regular = df_irregular.convert_frequency(freq="D")
```

### Filling Missing Values

**Default** (forward + backward fill):
```python
df_filled = df_regular.fill_missing_values()
```

**Constant fill** (e.g., for demand forecasting):
```python
df_filled = df_regular.fill_missing_values(method="constant", value=0.0)
```

**Note**: Most AutoGluon models handle NaNs natively.

## Forecast Accuracy Evaluation

### Train-Test Split

```python
prediction_length = 48
full_data = TimeSeriesDataFrame.from_path("https://autogluon.s3.amazonaws.com/datasets/timeseries/electricity_small/test.csv")
train_data, test_data = full_data.train_test_split(prediction_length)
```

- `train_data`: Historical data (last `prediction_length` steps removed)
- `test_data`: Full data (historical + forecast horizon)

### Evaluate Performance

```python
predictor = TimeSeriesPredictor(prediction_length=prediction_length, eval_metric="MASE")
predictor.fit(train_data, hyperparameters={"SeasonalNaive": {}, "RecursiveTabular": {}, "Chronos": {}})
predictor.evaluate(test_data)
```

**Evaluation process**:
1. Hold out last `prediction_length` values
2. Generate forecast for held-out period
3. Compute `eval_metric` on forecast vs. actual
4. Average scores across all time series

**Key**: `evaluate()` returns score for best model (by validation performance). Use `leaderboard()` for all models:

```python
predictor.leaderboard(test_data)
```

See [Forecasting Evaluation Metrics](forecasting-metrics.md) for details.

### Backtesting (Multiple Cutoffs)

More accurate evaluation using multiple forecast windows:

```python
num_test_windows = 3
train_data, test_data = full_data.train_test_split(num_test_windows * prediction_length)
```

# Backtesting & Model Selection

## Backtesting with Multiple Windows

```python
num_test_windows = 3
train_data, test_data = full_data.train_test_split(num_test_windows * prediction_length)

predictor = TimeSeriesPredictor(prediction_length=prediction_length, eval_metric="MASE")
predictor.fit(train_data, hyperparameters={"SeasonalNaive": {}, "RecursiveTabular": {}, "Chronos": {}})
```

Evaluate each window using `cutoff` parameter:

```python
for cutoff in range(-num_test_windows * prediction_length, 0, prediction_length):
    score = predictor.evaluate(test_data, cutoff=cutoff)
    print(f"Cutoff {cutoff}: score = {score}")
```

**Key**: `evaluate()` measures accuracy on `prediction_length` steps after `cutoff` index (default: `-1 * prediction_length`).

Compare all models across windows:

```python
leaderboards = []
for cutoff in range(-num_test_windows * prediction_length, 0, prediction_length):
    lb = predictor.leaderboard(test_data, cutoff=cutoff)
    lb["cutoff"] = cutoff
    leaderboards.append(lb)

scores_per_window = pd.concat(leaderboards).pivot(index="model", columns="cutoff", values="score_test")
```

## Visualizing Backtest Predictions

```python
predictions_per_window = predictor.backtest_predictions(test_data, num_val_windows=num_test_windows)
targets_per_window = predictor.backtest_targets(test_data, num_val_windows=num_test_windows)

# Plot all predictions
all_predictions = pd.concat(predictions_per_window)
predictor.plot(test_data, all_predictions, max_history_length=300, item_ids=item_ids)
```

## Internal Validation

AutoGluon automatically splits `train_data` into train/validation sets. Best validation model is used for prediction.

**Default**: Single validation window (last `prediction_length` steps).

**Multiple windows** (reduces overfitting, increases training time):
```python
predictor.fit(train_data, num_val_windows=3)
```
**Requirement**: Series length ≥ `(num_val_windows + 1) * prediction_length`

**Custom validation set**:
```python
predictor.fit(train_data=train_data, tuning_data=my_validation_dataset)
```

**View validation results**:
```python
predictor.leaderboard()  # Check 'score_val' column
val_predictions = predictor.backtest_predictions()
val_targets = predictor.backtest_targets()
```

## Available Forecasting Models

### Local Models
Simple statistical models fit separately per time series. Strong baselines for trend/seasonality.

- `ETS`
- `AutoARIMA`
- `Theta`
- `SeasonalNaive`

**Note**: Refit from scratch for new time series not in training set.

### Global Models
Single model trained on all time series.

**Neural networks** (via [GluonTS](https://ts.gluon.ai/stable/)):
- `DeepAR`
- `PatchTST`
- `DLinear`
- `TemporalFusionTransformer`
- `Chronos` (pre-trained zero-shot)

**Tabular models** (convert forecasting to regression):
- `RecursiveTabular`
- `DirectTabular`

### Ensemble Models
Combines predictions from all models.

**Default**: `WeightedEnsemble` (disable with `enable_ensemble=False` in `fit()`).

See [Forecasting Model Zoo](forecasting-model-zoo.md) for hyperparameters and details.

# Configuration & Hyperparameter Tuning

## Basic Configuration with Presets

```python
predictor = TimeSeriesPredictor(...)
predictor.fit(train_data, presets="medium_quality")
```

### Available Presets

| Preset | Description | Fit Time |
|--------|-------------|----------|
| `medium_quality` | Statistical models + fast tree-based + Chronos-2 (small) + Toto-2 (4m) | 1x |
| `high_quality` | Deep learning + ML + statistical + pretrained models | 3x |
| `best_quality` | Same as `high_quality` + more cross-validation windows | 6x |

**Note**: `best_quality` more accurate for datasets with <50 time series.

See [presets](https://github.com/autogluon/autogluon/blob/stable/timeseries/src/autogluon/timeseries/configs/predictor_presets.py) and [models per preset](https://github.com/autogluon/autogluon/blob/stable/timeseries/src/autogluon/timeseries/configs/hyperparameter_presets.py) in source code.

### Time Limit

```python
predictor.fit(train_data, time_limit=60 * 60)  # seconds
```

**Default**: Trains until all models complete (no time limit).

## Manual Model Configuration

Override presets with `hyperparameters`:

```python
predictor.fit(
    train_data,
    hyperparameters={
        "DeepAR": {},  # default hyperparameters
        "Theta": [
            {"decomposition_type": "additive"},
            {"seasonal_period": 1},  # disable seasonality
        ],
    }
)
```

**Result**: Trains 3 models (1 DeepAR + 2 Theta variants).

### Exclude Models from Presets

```python
predictor.fit(
    train_data,
    presets="high_quality",
    excluded_model_types=["AutoETS", "AutoARIMA"],
)
```

See [Forecasting Model Zoo](forecasting-model-zoo.md) for all models and hyperparameters.

## Hyperparameter Tuning (HPO)

Define search spaces for automatic hyperparameter optimization:

```python
from autogluon.common import space

predictor.fit(
    train_data,
    hyperparameters={
        "DeepAR": {
            "hidden_size": space.Int(20, 100),
            "dropout_rate": space.Categorical(0.1, 0.3),
        },
    },
    hyperparameter_tune_kwargs="auto",  # 10 trials by default
    enable_ensemble=False,
)
```

**Backend**: Ray Tune for GluonTS models, random search for others.

### Custom HPO Configuration

```python
predictor.fit(
    train_data,
    hyperparameter_tune_kwargs={
        "num_trials": 20,
        "scheduler": "local",  # only supported option
        "searcher": "random",  # only supported option
    },
)
```

**Required keys**:
- `num_trials`: int, configurations per model
- `searcher`: `"random"` (only option)
- `scheduler`: `"local"` (only option)

**Warning**: HPO significantly increases training time with often modest performance gains.