# Condensed: Forecasting Time Series - In Depth

Summary: This tutorial covers implementing time series forecasting with AutoGluon, focusing on three key areas: (1) core probabilistic forecasting concepts and handling static features, (2) working with covariates (known and past) and holiday features, including data formatting requirements and missing value handling, and (3) model configuration and tuning. It provides code examples for implementing various forecasting models (local, global, and ensemble), managing irregular data, evaluating forecasts through train-test splits and backtesting, and configuring hyperparameters. The tutorial demonstrates how to use different quality presets, customize model selection, and perform hyperparameter optimization using Ray Tune, making it particularly useful for tasks involving time series prediction, model evaluation, and performance optimization.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed version focusing on key implementation details and concepts:

# Forecasting Time Series - Core Concepts and Implementation

## Probabilistic Time Series Forecasting

### Key Components
- **Time Series**: Sequential measurements at regular intervals
- **Forecast Horizon**: Future period to predict, set via `prediction_length`
- **Output Types**:
  1. Mean forecast (expected values)
  2. Quantile forecasts (distribution percentiles)

```python
# Custom quantile configuration
predictor = TimeSeriesPredictor(quantile_levels=[0.05, 0.5, 0.95])
```

## Working with Additional Information

### Static Features
Static features are time-independent attributes for each time series.

#### Implementation Example:
```python
# Load data and static features
df = pd.read_csv("path_to_timeseries_data.csv")
static_features_df = pd.read_csv("path_to_static_features.csv")

# Create TimeSeriesDataFrame with static features
train_data = TimeSeriesDataFrame.from_data_frame(
    df,
    id_column="item_id",
    timestamp_column="timestamp",
    static_features_df=static_features_df,
)

# Alternative: Attach static features to existing TimeSeriesDataFrame
train_data.static_features = static_features_df
```

### Static Feature Types
- **Categorical**: 
  - Datatypes: `object`, `string`, `category`
  - Example: domain, store_id, product_id
- **Continuous**: 
  - Datatypes: `int`, `float`
  - Example: numerical attributes

#### Converting Feature Types:
```python
# Convert numeric to categorical
train_data.static_features["store_id"] = train_data.static_features["store_id"].astype("category")
```

### Important Notes
1. Static features must include all `item_id`s present in training data
2. Prediction data must include same static features with matching column names/types
3. Non-supported datatypes are ignored

### Basic Training
```python
predictor = TimeSeriesPredictor(prediction_length=14).fit(train_data)
```

This condensed version maintains all critical implementation details while removing redundant explanations and focusing on practical code examples and configurations.

Here's the condensed version focusing on key implementation details and practices:

# Time Series Covariates and Data Formatting in AutoGluon

## Covariates Implementation

### Types of Covariates
1. **Known Covariates**
   - Known for entire forecast horizon
   - Examples: holidays, day of week, promotions
   
2. **Past Covariates**
   - Known only up to forecast start
   - Examples: sales of other products, temperature, transformed target series

### Implementation Example
```python
# Create covariates
train_data["log_target"] = np.log(train_data["target"])  # past covariate
train_data["weekend"] = timestamps.weekday.isin([5, 6]).astype(float)  # known covariate

# Initialize predictor with covariates
predictor = TimeSeriesPredictor(
    prediction_length=14,
    target="target",
    known_covariates_names=["weekend"]
).fit(train_data)
```

### Prediction with Known Covariates
```python
# Generate future known covariates
future_index = get_forecast_horizon_index_ts_dataframe(train_data, prediction_length=14)
future_timestamps = future_index.get_level_values("timestamp")
known_covariates = pd.DataFrame(index=future_index)
known_covariates["weekend"] = future_timestamps.weekday.isin([5, 6]).astype(float)

# Predict
predictions = predictor.predict(train_data, known_covariates=known_covariates)
```

## Holiday Features Implementation

### Creating Holiday Features
```python
def add_holiday_features(ts_df, country_holidays, 
                        include_individual_holidays=True,
                        include_holiday_indicator=True):
    ts_df = ts_df.copy()
    timestamps = ts_df.index.get_level_values("timestamp")
    country_holidays_df = pd.get_dummies(pd.Series(country_holidays)).astype(float)
    holidays_df = country_holidays_df.reindex(timestamps.date).fillna(0)
    
    if include_individual_holidays:
        ts_df[holidays_df.columns] = holidays_df.values
    if include_holiday_indicator:
        ts_df["Holiday"] = holidays_df.max(axis=1).values
    return ts_df
```

### Using Holiday Features
```python
# Add holidays to training data
train_data_with_holidays = add_holiday_features(train_data, country_holidays)

# Initialize predictor with holiday features
holiday_columns = train_data_with_holidays.columns.difference(train_data.columns)
predictor = TimeSeriesPredictor(..., known_covariates_names=holiday_columns)
```

## Important Requirements

### Data Length Requirements
- Minimum length for default settings:
  ```
  length >= max(prediction_length + 1, 5) + prediction_length
  ```
- With advanced configuration:
  ```
  length >= max(prediction_length + 1, 5) + prediction_length + (num_val_windows - 1) * val_step_size
  ```

### Known Covariates Requirements
- Must include all columns listed in `predictor.known_covariates_names`
- `item_id` index must include all training data item ids
- `timestamp` index must cover `prediction_length` steps into future

Note: Time series in the dataset can have different lengths.

Here's the condensed version focusing on key implementation details and practices:

# Handling Irregular Data and Missing Values

## Irregular Time Series Data
```python
# Handle irregular data by specifying frequency during predictor creation
predictor = TimeSeriesPredictor(..., freq="D").fit(df_irregular)

# Manually convert irregular data to regular frequency
df_regular = df_irregular.convert_frequency(freq="D")
```

## Missing Value Handling
```python
# Default fill (forward + backward filling)
df_filled = df_regular.fill_missing_values()

# Constant fill (useful for demand forecasting)
df_filled = df_regular.fill_missing_values(method="constant", value=0.0)
```

# Forecast Evaluation

## Train-Test Split
```python
prediction_length = 48
train_data, test_data = data.train_test_split(prediction_length)

# Evaluate predictor
predictor = TimeSeriesPredictor(prediction_length=prediction_length, eval_metric="MASE")
predictor.fit(train_data)
predictor.evaluate(test_data)
```

## Multi-window Backtesting
```python
from autogluon.timeseries.splitter import ExpandingWindowSplitter

splitter = ExpandingWindowSplitter(prediction_length=prediction_length, num_val_windows=3)
for window_idx, (train_split, val_split) in enumerate(splitter.split(test_data)):
    score = predictor.evaluate(val_split)
```

## Validation Strategies

### Default Validation
- Uses last `prediction_length` timesteps as validation set
- Evaluates models on validation set to select best performer

### Multiple Validation Windows
```python
predictor.fit(train_data, num_val_windows=3)
```

### Custom Validation
```python
predictor.fit(train_data=train_data, tuning_data=my_validation_dataset)
```

## Important Notes:
1. Evaluation always occurs on last `prediction_length` timesteps
2. Multiple validation windows require time series length ≥ `(num_val_windows + 1) * prediction_length`
3. Multi-window backtesting provides more accurate performance estimates but requires longer time series
4. Increasing `num_val_windows` reduces overfitting but increases training time proportionally

Here's the condensed tutorial focusing on key implementation details and concepts:

# AutoGluon Forecasting Models and Configuration

## Available Forecasting Models

### Three Categories:
1. **Local Models**
   - Fit separately to each time series
   - Examples: ETS, AutoARIMA, Theta, SeasonalNaive
   - Good as baselines

2. **Global Models**
   - Learn from entire training set
   - Neural network models from GluonTS:
     - DeepAR, PatchTST, DLinear, TemporalFusionTransformer
   - Tabular models: RecursiveTabular, DirectTabular
   - Pre-trained models like Chronos

3. **Ensemble Models**
   - WeightedEnsemble (enabled by default)
   - Can be disabled with `enable_ensemble=False`

## TimeSeriesPredictor Configuration

### 1. Basic Configuration with Presets

```python
predictor.fit(train_data, presets="medium_quality")
```

Preset Options:
- `fast_training`: Simple models, quick training (0.5x time)
- `medium_quality`: Base models + TFT + Chronos-Bolt (1x time)
- `high_quality`: Advanced models, longer training (3x time)
- `best_quality`: More cross-validation, best for <50 series (6x time)

Time limit setting:
```python
predictor.fit(train_data, time_limit=60*60)  # in seconds
```

### 2. Manual Model Configuration

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
    excluded_model_types=["AutoETS", "AutoARIMA"]
)
```

### 3. Hyperparameter Tuning

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
    enable_ensemble=False
)
```

Custom HPO configuration:
```python
hyperparameter_tune_kwargs={
    "num_trials": 20,
    "scheduler": "local",
    "searcher": "random",
}
```

Important Notes:
- HPO uses Ray Tune for deep learning models
- Random search for other models
- HPO increases training time significantly
- Often provides modest performance gains