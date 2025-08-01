# Condensed: ```python

Summary: This tutorial demonstrates AutoGluon TimeSeriesPredictor for time series forecasting with covariates and static features. It covers: (1) implementing time series forecasting with static features, known covariates, and holiday indicators; (2) handling irregular data, missing values, and proper data formatting; and (3) configuring models with different presets and hyperparameter tuning. Key functionalities include creating TimeSeriesDataFrame objects, adding custom covariates, generating future prediction frames, evaluating forecast accuracy, and selecting from local models (ETS, ARIMA), global models (DeepAR, PatchTST), and ensemble approaches for optimal forecasting performance.

*This is a condensed version that preserves essential implementation details and context.*

# AutoGluon Time Series: Working with Covariates and Static Features

## Setup

```python
# Install dependencies
!pip install uv
!uv pip install -q autogluon.timeseries --system
!uv pip uninstall -q torchaudio torchvision torchtext --system # fix incompatible package versions on Colab

import warnings
warnings.filterwarnings(action="ignore")

import pandas as pd
import numpy as np
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
```

## Loading Data with Static Features

```python
# Load time series data
df = pd.read_csv("https://autogluon.s3.amazonaws.com/datasets/timeseries/m4_daily_subset/train.csv")

# Load static features
static_features_df = pd.read_csv("https://autogluon.s3.amazonaws.com/datasets/timeseries/m4_daily_subset/metadata.csv")

# Create TimeSeriesDataFrame with static features
train_data = TimeSeriesDataFrame.from_data_frame(
    df,
    id_column="item_id",
    timestamp_column="timestamp",
    static_features_df=static_features_df,
)

# Alternative: attach static features to existing TimeSeriesDataFrame
# train_data.static_features = static_features_df
```

## Adding Covariates

```python
# Add log-transformed target as past covariate
train_data["log_target"] = np.log(train_data["target"])

# Add weekend indicator as known covariate
WEEKEND_INDICES = [5, 6]
timestamps = train_data.index.get_level_values("timestamp")
train_data["weekend"] = timestamps.weekday.isin(WEEKEND_INDICES).astype(float)
```

## Training with Covariates

```python
predictor = TimeSeriesPredictor(
    prediction_length=14,
    target="target",
    known_covariates_names=["weekend"],
).fit(train_data)
```

## Making Predictions with Known Covariates

```python
# Generate future dataframe for known covariates
predictor = TimeSeriesPredictor(prediction_length=14, freq=train_data.freq)
known_covariates = predictor.make_future_data_frame(train_data)
known_covariates["weekend"] = known_covariates["timestamp"].dt.weekday.isin(WEEKEND_INDICES).astype(float)

# Make predictions with known covariates
predictions = predictor.predict(train_data, known_covariates=known_covariates)
```

## Working with Holiday Features

```python
!pip install -q holidays
import holidays
import datetime

# Get country holidays
timestamps = train_data.index.get_level_values("timestamp")
country_holidays = holidays.country_holidays(
    country="DE",  # select appropriate country/region
    years=range(timestamps.min().year, timestamps.max().year + 1),
)

# Alternative: define custom holidays
custom_holidays = {
    datetime.date(1995, 1, 29): "Superbowl",
    datetime.date(1995, 11, 29): "Black Friday",
    # Add more dates as needed
}

# Function to add holiday features
def add_holiday_features(
    ts_df: TimeSeriesDataFrame,
    country_holidays: dict,
    include_individual_holidays: bool = True,
    include_holiday_indicator: bool = True,
) -> TimeSeriesDataFrame:
    """Add holiday indicator columns to a TimeSeriesDataFrame."""
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

# Add holiday features to training data
train_data_with_holidays = add_holiday_features(train_data, country_holidays)

# Train with holiday features
holiday_columns = train_data_with_holidays.columns.difference(train_data.columns)
predictor = TimeSeriesPredictor(
    prediction_length=14,
    target="target",
    known_covariates_names=holiday_columns
).fit(train_data_with_holidays)

# Generate future known covariates with holidays
known_covariates = predictor.make_future_data_frame(train_data)
known_covariates = add_holiday_features(known_covariates, country_holidays)

# Make predictions with holiday features
predictions = predictor.predict(train_data_with_holidays, known_covariates=known_covariates)
```

**Important Notes:**
- Known covariates must include all columns listed in `predictor.known_covariates_names`
- The `item_id` index must include all item IDs present in training data
- The `timestamp` index must include values for `prediction_length` time steps into the future
- Check the [Forecasting Model Zoo](forecasting-model-zoo.md) for models supporting static features and covariates

# Data Format and Handling in TimeSeriesPredictor

## Data Length Requirements

TimeSeriesPredictor requires time series of sufficient length:
- With default settings: At least some time series must have length `>= max(prediction_length + 1, 5) + prediction_length`
- With custom validation settings: Length must be `>= max(prediction_length + 1, 5) + prediction_length + (num_val_windows - 1) * val_step_size`

Time series in the dataset can have different lengths.

## Handling Irregular Data and Missing Values

For irregular time series data:

```python
# Specify frequency when creating predictor
predictor = TimeSeriesPredictor(..., freq="D").fit(df_irregular)

# Or manually convert frequency before training
df_regular = df_irregular.convert_frequency(freq="D")
```

For handling missing values:
```python
# Default fill (forward + backward)
df_filled = df_regular.fill_missing_values()

# Custom fill (e.g., for demand forecasting)
df_filled = df_regular.fill_missing_values(method="constant", value=0.0)
```

## Evaluating Forecast Accuracy

1. Split data into train and test sets:
```python
train_data, test_data = data.train_test_split(prediction_length)
```

2. Train and evaluate:
```python
predictor = TimeSeriesPredictor(prediction_length=prediction_length, eval_metric="MASE").fit(train_data)
predictor.evaluate(test_data)
```

The evaluation process:
1. Holds out the last `prediction_length` values of each time series
2. Generates forecasts for the held-out period
3. Compares forecasts with actual values using the specified metric
4. Averages scores across all time series

## Validation Process

By default, AutoGluon uses the last `prediction_length` time steps of each time series for validation.

For more robust validation:
```python
predictor.fit(train_data, num_val_windows=3)
```

This creates multiple validation windows but requires longer time series and increases training time.

You can also provide a custom validation set:
```python
predictor.fit(train_data=train_data, tuning_data=my_validation_dataset)
```

# AutoGluon Forecasting Models and TimeSeriesPredictor Configuration

## Available Forecasting Models

AutoGluon offers three categories of forecasting models:

### Local Models
- Simple statistical models fit separately to each time series
- Examples: `ETS`, `AutoARIMA`, `Theta`, `SeasonalNaive`
- For new time series, these models are fit from scratch

### Global Models
- Machine learning algorithms that learn from multiple time series
- Neural network models from GluonTS library:
  - `DeepAR`, `PatchTST`, `DLinear`, `TemporalFusionTransformer`
- Pre-trained zero-shot models like Chronos
- Tabular models: `RecursiveTabular` and `DirectTabular` (convert forecasting to regression)

### Ensemble Models
- `WeightedEnsemble` combines predictions from other models
- Enabled by default, can be disabled with `enable_ensemble=False`

## TimeSeriesPredictor Configuration Options

### Basic Configuration with Presets

```python
predictor = TimeSeriesPredictor(...)
predictor.fit(train_data, presets="medium_quality")
```

Available presets:
- `fast_training`: Simple models, quick training (0.5x time)
- `medium_quality`: Adds TFT and Chronos-Bolt (1x time)
- `high_quality`: More powerful models (3x time)
- `best_quality`: More cross-validation windows (6x time)

Control training duration with `time_limit`:
```python
predictor.fit(train_data, time_limit=60*60)  # in seconds
```

### Manual Model Configuration

Specify models and parameters:
```python
predictor.fit(
    train_data,
    hyperparameters={
        "DeepAR": {},
        "Theta": [
            {"decomposition_type": "additive"},
            {"seasonal_period": 1},
        ],
    }
)
```

Exclude specific models:
```python
predictor.fit(
    train_data,
    presets="high_quality",
    excluded_model_types=["AutoETS", "AutoARIMA"],
)
```

### Hyperparameter Tuning

Define search spaces for model parameters:
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
    hyperparameter_tune_kwargs="auto",
    enable_ensemble=False,
)
```

Custom tuning configuration:
```python
predictor.fit(
    train_data,
    hyperparameter_tune_kwargs={
        "num_trials": 20,
        "scheduler": "local",
        "searcher": "random",
    }
)
```

**Note:** HPO significantly increases training time but often provides modest performance gains.