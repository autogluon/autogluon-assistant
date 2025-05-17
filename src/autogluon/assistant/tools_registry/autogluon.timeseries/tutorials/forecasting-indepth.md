Summary: This tutorial covers implementing time series forecasting with AutoGluon, focusing on three key areas: (1) core probabilistic forecasting concepts and handling static features, (2) working with covariates (known and past) and holiday features, including data formatting requirements and missing value handling, and (3) model configuration and tuning. It provides code examples for implementing various forecasting models (local, global, and ensemble), managing irregular data, evaluating forecasts through train-test splits and backtesting, and configuring hyperparameters. The tutorial demonstrates how to use different quality presets, customize model selection, and perform hyperparameter optimization using Ray Tune, making it particularly useful for tasks involving time series prediction, model evaluation, and performance optimization.

# Forecasting Time Series - In Depth

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/autogluon/autogluon/blob/master/docs/tutorials/timeseries/forecasting-indepth.ipynb)
[![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/autogluon/autogluon/blob/master/docs/tutorials/timeseries/forecasting-indepth.ipynb)


This tutorial provides an in-depth overview of the time series forecasting capabilities in AutoGluon.
Specifically, we will cover:

- What is probabilistic time series forecasting?
- Forecasting time series with additional information
- What data format is expected by `TimeSeriesPredictor`?
- How to evaluate forecast accuracy?
- Which forecasting models are available in AutoGluon?
- What functionality does `TimeSeriesPredictor` offer?
    - Basic configuration with `presets` and `time_limit`
    - Manually selecting what models to train
    - Hyperparameter tuning

This tutorial assumes that you are familiar with the contents of [Forecasting Time Series - Quick Start](forecasting-quick-start.ipynb).

## What is probabilistic time series forecasting?
A time series is a sequence of measurements made at regular intervals.
The main objective of time series forecasting is to predict the future values of a time series given the past observations.
A typical example of this task is demand forecasting.
For example, we can represent the number of daily purchases of a certain product as a time series.
The goal in this case could be predicting the demand for each of the next 14 days (i.e., the forecast horizon) given the historical purchase data.
In AutoGluon, the `prediction_length` argument of the `TimeSeriesPredictor` determines the length of the forecast horizon.

![Main goal of forecasting is to predict the future values of a time series given the past observations.](https://autogluon-timeseries-datasets.s3.us-west-2.amazonaws.com/public/figures/forecasting-indepth1.png)

The objective of forecasting could be to predict future values of a given time series, as well as establishing prediction intervals within which the future values will likely lie.
In AutoGluon, the `TimeSeriesPredictor` generates two types of forecasts:

- **mean forecast** represents the expected value of the time series at each time step in the forecast horizon.
- **quantile forecast** represents the quantiles of the forecast distribution.
For example, if the `0.1` quantile (also known as P10, or the 10th percentile) is equal to `x`, it means that the time series value is predicted to be below `x` 10% of the time. As another example, the `0.5` quantile (P50) corresponds to the median forecast.
Quantiles can be used to reason about the range of possible outcomes.
For instance, by the definition of the quantiles, the time series is predicted to be between the P10 and P90 values with 80% probability.


![Mean and quantile (P10 and P90) forecasts.](https://autogluon-timeseries-datasets.s3.us-west-2.amazonaws.com/public/figures/forecasting-indepth2.png)

By default, the `TimeSeriesPredictor` outputs the quantiles `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]`. Custom quantiles can be provided with the `quantile_levels` argument

```python
predictor = TimeSeriesPredictor(quantile_levels=[0.05, 0.5, 0.95])
```

## Forecasting time series with additional information
In real-world forecasting problems we often have access to additional information, beyond just the raw time series values.
AutoGluon supports two types of such additional information: static features and time-varying covariates.

### Static features
Static features are the time-independent attributes (metadata) of a time series.
These may include information such as:

- location, where the time series was recorded (country, state, city)
- fixed properties of a product (brand name, color, size, weight)
- store ID or product ID

Providing this information may, for instance, help forecasting models generate similar demand forecasts for stores located in the same city.

In AutoGluon, static features are stored as an attribute of a `TimeSeriesDataFrame` object.
As an example, let's have a look at the M4 Daily dataset.


```python
# We use uv for faster installation
!pip install uv
!uv pip install -q autogluon.timeseries --system
!uv pip uninstall -q torchaudio torchvision torchtext --system # fix incompatible package versions on Colab
```


```python
import warnings
warnings.filterwarnings(action="ignore")

```


```python
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

```

We download a subset of 100 time series from the M4 Daily dataset.


```python
df = pd.read_csv("https://autogluon.s3.amazonaws.com/datasets/timeseries/m4_daily_subset/train.csv")
df.head()

```

We also load the corresponding static features.
In the M4 Daily dataset, there is a single categorical static feature that denotes the domain of origin for each time series.



```python
static_features_df = pd.read_csv("https://autogluon.s3.amazonaws.com/datasets/timeseries/m4_daily_subset/metadata.csv")
static_features_df.head()

```

AutoGluon expects static features as a pandas.DataFrame object. The `item_id` column indicates which item (=individual time series) in `df` each row of `static_features` corresponds to.

We can now create a `TimeSeriesDataFrame` that contains both the time series values and the static features.


```python
train_data = TimeSeriesDataFrame.from_data_frame(
    df,
    id_column="item_id",
    timestamp_column="timestamp",
    static_features_df=static_features_df,
)
train_data.head()

```

We can validate that `train_data` now also includes the static features using the `.static_features` attribute


```python
train_data.static_features.head()

```

Alternatively, we can attach static features to an existing `TimeSeriesDataFrame` by assigning the `.static_features` attribute


```python
train_data.static_features = static_features_df

```


If `static_features` doesn't contain some `item_id`s that are present in `train_data`, an exception will be raised.

Now, when we fit the predictor, all models that support static features will automatically use the static features included in `train_data`.

```python
predictor = TimeSeriesPredictor(prediction_length=14).fit(train_data)
```

```
...
Following types of static features have been inferred:
	categorical: ['domain']
	continuous (float): []
...
```

This message confirms that column `'domain'` was interpreted as a categorical feature.
In general, AutoGluon-TimeSeries supports two types of static features:

- `categorical`: columns of dtype `object`, `string` and `category` are interpreted as discrete categories
- `continuous`: columns of dtype `int` and `float` are interpreted as continuous (real-valued) numbers
- columns with other dtypes are ignored

To override this logic, we need to manually change the columns dtype.
For example, suppose the static features data frame contained an integer-valued column `"store_id"`.

```python
train_data.static_features["store_id"] = list(range(len(train_data.item_ids)))
```

By default, this column will be interpreted as a continuous number.
We can force AutoGluon to interpret it a a categorical feature by changing the dtype to `category`.

```python
train_data.static_features["store_id"] = train_data.static_features["store_id"].astype("category")
```

**Note:** If training data contained static features, the predictor will expect that data passed to `predictor.predict()`, `predictor.leaderboard()`, and `predictor.evaluate()` also includes static features with the same column names and data types.


### Time-varying covariates
Covariates are the time-varying features that may influence the target time series.
They are sometimes also referred to as dynamic features, exogenous regressors, or related time series.
AutoGluon supports two types of covariates:

- *known* covariates that are known for the entire forecast horizon, such as
    - holidays
    - day of the week, month, year
    - promotions

- *past* covariates that are only known up to the start of the forecast horizon, such as
    - sales of other products
    - temperature, precipitation
    - transformed target time series


![Target time series with one past covariate and one known covariate.](https://autogluon-timeseries-datasets.s3.us-west-2.amazonaws.com/public/figures/forecasting-indepth5.png)

In AutoGluon, both `known_covariates` and `past_covariates` are stored as additional columns in the `TimeSeriesDataFrame`.

We will again use the M4 Daily dataset as an example and generate both types of covariates:

- a `past_covariate` equal to the logarithm of the target time series:
- a `known_covariate` that equals to 1 if a given day is a weekend, and 0 otherwise.


```python
import numpy as np
train_data["log_target"] = np.log(train_data["target"])

WEEKEND_INDICES = [5, 6]
timestamps = train_data.index.get_level_values("timestamp")
train_data["weekend"] = timestamps.weekday.isin(WEEKEND_INDICES).astype(float)

train_data.head()

```

When creating the TimeSeriesPredictor, we specify that the column `"target"` is our prediction target, and the
column `"weekend"` contains a covariate that will be known at prediction time.

```python
predictor = TimeSeriesPredictor(
    prediction_length=14,
    target="target",
    known_covariates_names=["weekend"],
).fit(train_data)
```

Predictor will automatically interpret the remaining columns (except target and known covariates) as past covariates.
This information is logged during fitting:

```
...
Provided dataset contains following columns:
	target:           'target'
	known covariates: ['weekend']
	past covariates:  ['log_target']
...
```

Finally, to make predictions, we generate the known covariates for the forecast horizon


```python
from autogluon.timeseries.utils.forecast import get_forecast_horizon_index_ts_dataframe

future_index = get_forecast_horizon_index_ts_dataframe(train_data, prediction_length=14)
future_timestamps = future_index.get_level_values("timestamp")
known_covariates = pd.DataFrame(index=future_index)
known_covariates["weekend"] = future_timestamps.weekday.isin(WEEKEND_INDICES).astype(float)

known_covariates.head()

```

Note that `known_covariates` must satisfy the following conditions:

- The columns must include all columns listed in ``predictor.known_covariates_names``
- The ``item_id`` index must include all item ids present in ``train_data``
- The ``timestamp`` index must include the values for ``prediction_length`` many time steps into the future from the end of each time series in ``train_data``

If `known_covariates` contain more information than necessary (e.g., contain additional columns, item_ids, or timestamps),
AutoGluon will automatically select the necessary rows and columns.

Finally, we pass the `known_covariates` to the `predict` function to generate predictions

```python
predictor.predict(train_data, known_covariates=known_covariates)
```

The list of models that support static features and covariates is available in [Forecasting Model Zoo](forecasting-model-zoo.md).

### Holidays
Another popular example of `known_covariates` are holiday features. In this section we describe how to add holiday features to a time series dataset and use them in AutoGluon.

First, we need to define a dictionary with dates in `datetime.date` format as keys and holiday names as values. 
We can easily generate such a dictionary using the [`holidays`](https://pypi.org/project/holidays/) Python package.


```python
!pip install -q holidays
```

Here we use German holidays for demonstration purposes only. Make sure to define a holiday calendar that matches your country/region!


```python
import holidays

timestamps = train_data.index.get_level_values("timestamp")
country_holidays = holidays.country_holidays(
    country="DE",  # make sure to select the correct country/region!
    # Add + 1 year to make sure that holidays are initialized for the forecast horizon
    years=range(timestamps.min().year, timestamps.max().year + 1),
)
# Convert dict to pd.Series for pretty visualization
pd.Series(country_holidays).sort_index().head()
```

Alternatively, we can manually define a dictionary with custom holidays.


```python
import datetime

# must cover the full train time range + forecast horizon
custom_holidays = {
    datetime.date(1995, 1, 29): "Superbowl",
    datetime.date(1995, 11, 29): "Black Friday",
    datetime.date(1996, 1, 28): "Superbowl",
    datetime.date(1996, 11, 29): "Black Friday",
    # ...
}
```

Next, we define a method that adds holiday features as columns to a `TimeSeriesDataFrame`.


```python
def add_holiday_features(
    ts_df: TimeSeriesDataFrame,
    country_holidays: dict,
    include_individual_holidays: bool = True,
    include_holiday_indicator: bool = True,
) -> TimeSeriesDataFrame:
    """Add holiday indicator columns to a TimeSeriesDataFrame."""
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

We can create a single indicator feature for all holidays.


```python
add_holiday_features(train_data, country_holidays, include_individual_holidays=False).head()
```

Or represent each holiday with a separate feature.


```python
train_data_with_holidays = add_holiday_features(train_data, country_holidays)
train_data_with_holidays.head()
```

Remember to add the names of holiday features as `known_covariates_names` when creating `TimeSeriesPredictor`.

```python
holiday_columns = train_data_with_holidays.columns.difference(train_data.columns)
predictor = TimeSeriesPredictor(..., known_covariates_names=holiday_columns).fit(train_data_with_holidays, ...)
```

At prediction time, we need to provide future holiday values as `known_covariates`.


```python
future_index = get_forecast_horizon_index_ts_dataframe(train_data, prediction_length=14)
future_timestamps = future_index.get_level_values("timestamp")
known_covariates = add_holiday_features(pd.DataFrame(index=future_index), country_holidays)
known_covariates.head()
```

```python
predictions = predictor.predict(train_data_with_holidays, known_covariates=known_covariates)
```

## What data format is expected by `TimeSeriesPredictor`?

AutoGluon expects that at least some time series in the training data are long enough to generate an internal validation set.

This means, at least some time series in `train_data` must have length `>= max(prediction_length + 1, 5) + prediction_length` when training with default settings
```python
predictor = TimeSeriesPredictor(prediction_length=prediction_length).fit(train_data)
```

If you use advanced configuration options, such as following,
```python
predictor = TimeSeriesPredictor(prediction_length=prediction_length).fit(train_data, num_val_windows=num_val_windows, val_step_size=val_step_size)
```
then at least some time series in `train_data` must have length `>= max(prediction_length + 1, 5) + prediction_length + (num_val_windows - 1) * val_step_size`.

Note that all time series in the dataset can have different lengths.


### Handling irregular data and missing values 
In some applications, like finance, data often comes with irregular measurements (e.g., no stock price is available for weekends or holidays) or missing values.

Here is an example of a dataset with an irregular time index:


```python
df_irregular = TimeSeriesDataFrame(
    pd.DataFrame(
        {
            "item_id": [0, 0, 0, 1, 1],
            "timestamp": ["2022-01-01", "2022-01-02", "2022-01-04", "2022-01-01", "2022-01-04"],
            "target": [1, 2, 3, 4, 5],
        }
    )
)
df_irregular

```

In such case, you can specify the desired frequency when creating the predictor using the `freq` argument.
```python
predictor = TimeSeriesPredictor(..., freq="D").fit(df_irregular)
```
Here we choose `freq="D"` to indicate that the filled index must have a daily frequency
(see [other possible choices in pandas documentation](https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases)).

AutoGluon will automatically convert the irregular data into daily frequency and deal with missing values.

--------
Alternatively, we can manually fill the gaps in the time index using the method [TimeSeriesDataFrame.convert_frequency()](../../api/autogluon.timeseries.TimeSeriesDataFrame.convert_frequency.rst).


```python
df_regular = df_irregular.convert_frequency(freq="D")
df_regular

```

We can verify that the index is now regular and has a daily frequency


```python
print(f"Data has frequency '{df_regular.freq}'")

```

Now the data contains missing values represented by `NaN`. Most time series models in AutoGluon can natively deal with missing values, so we can just pass data to the `TimeSeriesPredictor`.

Alternatively, we can manually fill the NaNs with an appropriate strategy using [TimeSeriesDataFrame.fill_missing_values()](../../api/autogluon.timeseries.TimeSeriesDataFrame.fill_missing_values.rst).
By default, missing values are filled with a combination of forward + backward filling.


```python
df_filled = df_regular.fill_missing_values()
df_filled
```

In some applications such as demand forecasting, missing values may correspond to zero demand. In this case constant fill is more appropriate.


```python
df_filled = df_regular.fill_missing_values(method="constant", value=0.0)
df_filled
```

## How to evaluate forecast accuracy?

To measure how accurately `TimeSeriesPredictor` can forecast unseen time series, we need to reserve some test data that won't be used for training.
This can be easily done using the `train_test_split` method of a `TimeSeriesDataFrame`:


```python
prediction_length = 48
data = TimeSeriesDataFrame.from_path("https://autogluon.s3.amazonaws.com/datasets/timeseries/m4_hourly_subset/train.csv")
train_data, test_data = data.train_test_split(prediction_length)

```

We obtained two `TimeSeriesDataFrame`s from our original data:
- `test_data` contains exactly the same data as the original `data` (i.e., it contains both historic data and the forecast horizon)
- In `train_data`, the last `prediction_length` time steps are removed from the end of each time series (i.e., it contains only historic data)


```python
import matplotlib.pyplot as plt
import numpy as np

item_id = "H1"
fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=[10, 4], sharex=True)
train_ts = train_data.loc[item_id]
test_ts = test_data.loc[item_id]
ax1.set_title("Train data (past time series values)")
ax1.plot(train_ts)
ax2.set_title("Test data (past + future time series values)")
ax2.plot(test_ts)
for ax in (ax1, ax2):
    ax.fill_between(np.array([train_ts.index[-1], test_ts.index[-1]]), test_ts.min(), test_ts.max(), color="C1", alpha=0.3, label="Forecast horizon")
plt.legend()
plt.show()

```

We can now use `train_data` to train the predictor, and `test_data` to obtain an estimate of its performance on unseen data.
```python
predictor = TimeSeriesPredictor(prediction_length=prediction_length, eval_metric="MASE").fit(train_data)
predictor.evaluate(test_data)
```

AutoGluon evaluates the performance of forecasting models by measuring how well their forecasts align with the actually observed time series.
For each time series in `test_data`, the predictor does the following:

1. Hold out the last `prediction_length` values of the time series.
2. Generate a forecast for the held out part of the time series, i.e., the forecast horizon.
3. Quantify how well the forecast matches the actually observed (held out) values of the time series using the `eval_metric`.

Finally, the scores are averaged over all time series in the dataset.

The crucial detail here is that `evaluate` always computes the score on the last `prediction_length` time steps of each time series.
The beginning of each time series (except the last `prediction_length` time steps) is only used to initialize the models before forecasting.

For more details about the evaluation metrics, see [Forecasting Evaluation Metrics](forecasting-metrics.md).

### Backtesting using multiple windows

We can more accurately estimate the performance using **backtest** (i.e., evaluate performance on multiple forecast horizons generated from the same time series).
This can be done using an `ExpandingWindowSplitter`.

```python
from autogluon.timeseries.splitter import ExpandingWindowSplitter

splitter = ExpandingWindowSplitter(prediction_length=prediction_length, num_val_windows=3)
for window_idx, (train_split, val_split) in enumerate(splitter.split(test_data)):
    score = predictor.evaluate(val_split)
    print(f"Window {window_idx}: score = {score}")
```

The `evaluate` method will measure the forecast accuracy using the last `prediction_length` time steps of each validation split as a hold-out set (marked in orange).

![MultiWindowSplitter splits each original time series into multiple evaluation instances. Forecast is evaluated on the last `prediction_length` timesteps (orange).](https://autogluon-timeseries-datasets.s3.us-west-2.amazonaws.com/public/figures/forecasting-indepth6.png)

Multi-window backtesting typically results in more accurate estimation of the forecast quality on unseen data.
However, this strategy decreases the amount of training data available for fitting models, so we recommend using single-window backtesting if the training time series are short.

### How does AutoGluon perform validation?
When we fit the predictor with `predictor.fit(train_data=train_data)`, under the hood AutoGluon further splits the original dataset `train_data` into train and validation parts.

Performance of different models on the validation set is evaluated using the `evaluate` method, just like described above.
The model that achieves the best validation score will be used for prediction in the end.

By default, the internal validation set contains a single window containing the last `prediction_length` time steps of each time series. We can increase the number of validation windows using the `num_val_windows` argument.

```python
predictor = TimeSeriesPredictor(...)
predictor.fit(train_data, num_val_windows=3)
```
This will reduce the likelihood of overfitting but will increase the training time approximately by a factor of `num_val_windows`.
Note that multiple validation windows can only be used if the time series in `train_data` have length of at least `(num_val_windows + 1) * prediction_length`.

Alternatively, a user can provide their own validation set to the `fit` method. In this case it's important to remember that the validation score is computed on the last `prediction_length` time steps of each time series.

```python
predictor.fit(train_data=train_data, tuning_data=my_validation_dataset)
```

## Which forecasting models are available in AutoGluon?
Forecasting models in AutoGluon can be divided into three broad categories: local, global, and ensemble models.

**Local models** are simple statistical models that are specifically designed to capture patterns such as trend or seasonality.
Despite their simplicity, these models often produce reasonable forecasts and serve as a strong baseline.
Some examples of available local models:

- `ETS`
- `AutoARIMA`
- `Theta`
- `SeasonalNaive`

If the dataset consists of multiple time series, we fit a separate local model to each time series — hence the name "local".
This means, if we want to make a forecast for a new time series that wasn't part of the training set, all local models will be fit from scratch for the new time series.

**Global models** are machine learning algorithms that learn a single model from the entire training set consisting of multiple time series.
Most global models in AutoGluon are provided by the [GluonTS](https://ts.gluon.ai/stable/) library.
These are neural-network algorithms implemented in PyTorch, such as:

- `DeepAR`
- `PatchTST`
- `DLinear`
- `TemporalFusionTransformer`

This category also includes pre-trained zero-shot forecasting models like [Chronos](forecasting-chronos.ipynb).

AutoGluon also offers two tabular global models `RecursiveTabular` and `DirectTabular`.
Under the hood, these models convert the forecasting task into a regression problem and use a [TabularPredictor](../../api/autogluon.tabular.TabularPredictor.rst) to fit regression algorithms like LightGBM.

Finally, an **ensemble** model works by combining predictions of all other models.
By default, `TimeSeriesPredictor` always fits a `WeightedEnsemble` on top of other models.
This can be disabled by setting `enable_ensemble=False` when calling the `fit` method.

For a list of tunable hyperparameters for each model, their default values, and other details see [Forecasting Model Zoo](forecasting-model-zoo.md).


## What functionality does `TimeSeriesPredictor` offer?
AutoGluon offers multiple ways to configure the behavior of a `TimeSeriesPredictor` that are suitable for both beginners and expert users.

### Basic configuration with `presets` and `time_limit`
We can fit `TimeSeriesPredictor` with different pre-defined configurations using the `presets` argument of the `fit` method.

```python
predictor = TimeSeriesPredictor(...)
predictor.fit(train_data, presets="medium_quality")
```

Higher quality presets usually result in better forecasts but take longer to train.
The following presets are available:

| Preset         | Description                                          | Use Cases                                                                                                                                               | Fit Time (Ideal) | 
| :------------- | :----------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------ | :--------------- | 
| `fast_training`  | Fit simple statistical and baseline models + fast tree-based models   | Fast to train but may not be very accurate   |  0.5x |
| `medium_quality` | Same models as in `fast_training` + deep learning model `TemporalFusionTransformer` + Chronos-Bolt (small)           | Good forecasts with reasonable training time         | 1x             |
| `high_quality`   | More powerful deep learning, machine learning, statistical and pretrained forecasting models   | Much more accurate than ``medium_quality``, but takes longer to train | 3x |
| `best_quality`   | Same models as in `high_quality`, more cross-validation windows | Typically more accurate than `high_quality`, especially for datasets with few (<50) time series | 6x             |

You can find more information about the [presets](https://github.com/autogluon/autogluon/blob/stable/timeseries/src/autogluon/timeseries/configs/presets_configs.py) and the [models includes in each preset](https://github.com/autogluon/autogluon/blob/stable/timeseries/src/autogluon/timeseries/models/presets.py#L109) in the AutoGluon source code.

Another way to control the training time is using the `time_limit` argument.

```python
predictor.fit(
    train_data,
    time_limit=60 * 60,  # total training time in seconds
)
```

If no `time_limit` is provided, the predictor will train until all models have been fit.


### Manually configuring models
Advanced users can override the presets and manually specify what models should be trained by the predictor using the `hyperparameters` argument.

```python
predictor = TimeSeriesPredictor(...)

predictor.fit(
    ...
    hyperparameters={
        "DeepAR": {},
        "Theta": [
            {"decomposition_type": "additive"},
            {"seasonal_period": 1},
        ],
    }
)
```

The above example will train three models:

* ``DeepAR`` with default hyperparameters
* ``Theta`` with additive seasonal decomposition (all other parameters set to their defaults)
* ``Theta`` with seasonality disabled (all other parameters set to their defaults)

You can also exclude certain models from the presets using the `excluded_model_type` argument.
```python
predictor.fit(
    ...
    presets="high_quality",
    excluded_model_types=["AutoETS", "AutoARIMA"],
)
```

For the full list of available models and the respective hyperparameters, see [Forecasting Model Zoo](forecasting-model-zoo.md).

### Hyperparameter tuning

Advanced users can define search spaces for model hyperparameters and let AutoGluon automatically determine the best configuration for the model.

```python
from autogluon.common import space

predictor = TimeSeriesPredictor()

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

This code will train multiple versions of the `DeepAR` model with 10 different hyperparameter configurations.
AutGluon will automatically select the best model configuration that achieves the highest validation score and use it for prediction.

Currently, HPO is based on Ray Tune for deep learning models from GluonTS, and random search for all other time series models.

We can change the number of random search trials per model by passing a dictionary as `hyperparameter_tune_kwargs`

```python
predictor.fit(
    ...
    hyperparameter_tune_kwargs={
        "num_trials": 20,
        "scheduler": "local",
        "searcher": "random",
    },
    ...
)
```

The `hyperparameter_tune_kwargs` dict must include the following keys:

- ``"num_trials"``: int, number of configurations to train for each tuned model
- ``"searcher"``: currently, the only supported option is ``"random"`` (random search).
- ``"scheduler"``: currently, the only supported option is ``"local"`` (all models trained on the same machine)

**Note:** HPO significantly increases the training time for most models, but often provides only modest performance gains.
