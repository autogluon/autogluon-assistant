Summary: This tutorial demonstrates the implementation of a NHITS (Neural Hierarchical Interpolation for Time Series) model within AutoGluon's framework. It provides implementation details for model class setup, data preprocessing, model fitting with GPU support, and data format conversion. The tutorial helps with tasks like handling missing values, integrating deep learning models into AutoGluon, managing time limits, and supporting various feature types (known covariates, past covariates, static features). Key functionalities covered include GPU acceleration, quantile-based predictions, automatic preprocessing, and compatibility with AutoGluon's ensemble capabilities. The implementation knowledge is particularly valuable for developers looking to integrate custom neural forecasting models into the AutoGluon ecosystem.

# Adding a custom time series forecasting model

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/autogluon/autogluon/blob/master/docs/tutorials/timeseries/advanced/forecasting-custom-model.ipynb)
[![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/autogluon/autogluon/blob/master/docs/tutorials/timeseries/advanced/forecasting-custom-model.ipynb)

This tutorial describes how to add a custom forecasting model that can be trained, hyperparameter-tuned, and ensembled alongside the [default forecasting models](../forecasting-model-zoo.html).

As an example, we will implement an AutoGluon wrapper for the [NHITS](https://nixtlaverse.nixtla.io/neuralforecast/models.nhits.html#nhits) model from the [NeuralForecast](https://github.com/Nixtla/NeuralForecast) library.

This tutorial consists of the following sections:
1. Implementing the model wrapper.
2. Loading & preprocessing the dataset used for model development.
3. Using the custom model in standalone mode.
4. Using the custom model inside the `TimeSeriesPredictor`.


```{warning}

This tutorial is designed for advanced AutoGluon users.

Custom model implementations rely heavily on the private of API of AutoGluon that might change over time. For this reason, it might be necessary to update your custom model implementations as you upgrade to new versions of AutoGluon.

```


```python
# We use uv for faster installation
!pip install uv
!uv pip install -q autogluon.timeseries --system
!uv pip uninstall -q torchaudio torchvision torchtext --system # fix incompatible package versions on Colab
```

First, we install the NeuralForecast library that contains the implementation of the custom model used in this tutorial.


```python
pip install -q neuralforecast==2.0
```

## Implement the custom model


To implement a custom model we need to create a subclass of the [`AbstractTimeSeriesModel`](https://github.com/autogluon/autogluon/blob/master/timeseries/src/autogluon/timeseries/models/abstract/abstract_timeseries_model.py) class. This subclass must implement two methods: `_fit` and `_predict`. For models that require a custom preprocessing logic (e.g., to handle missing values), we also need to implement the `preprocess` method.

Please have a look at the following code and read the comments to understand the different components of the custom model wrapper.


```python
import logging
import pprint
from typing import Optional, Tuple

import pandas as pd

from autogluon.timeseries import TimeSeriesDataFrame
from autogluon.timeseries.models.abstract import AbstractTimeSeriesModel
from autogluon.timeseries.utils.warning_filters import warning_filter

# Optional - disable annoying PyTorch-Lightning loggers
for logger_name in [
    "lightning.pytorch.utilities.rank_zero",
    "pytorch_lightning.accelerators.cuda",
    "lightning_fabric.utilities.seed",
]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)


class NHITSModel(AbstractTimeSeriesModel):
    """AutoGluon-compatible wrapper for the NHITS model from NeuralForecast."""

    # Set these attributes to ensure that AutoGluon passes correct features to the model
    _supports_known_covariates: bool = True
    _supports_past_covariates: bool = True
    _supports_static_features: bool = True

    def preprocess(
        self,
        data: TimeSeriesDataFrame,
        known_covariates: Optional[TimeSeriesDataFrame] = None,
        is_train: bool = False,
        **kwargs,
    ) -> Tuple[TimeSeriesDataFrame, Optional[TimeSeriesDataFrame]]:
        """Method that implements model-specific preprocessing logic.

        This method is called on all data that is passed to `_fit` and `_predict` methods.
        """
        # NeuralForecast cannot handle missing values represented by NaN. Therefore, we
        # need to impute them before the data is passed to the model. First, we
        # forward-fill and backward-fill all time series
        data = data.fill_missing_values()
        # Some time series might consist completely of missing values, so the previous
        # line has no effect on them. We fill them with 0.0
        data = data.fill_missing_values(method="constant", value=0.0)
        # Some models (e.g., Chronos) can natively handle NaNs - for them we don't need
        # to define a custom preprocessing logic
        return data, known_covariates

    def _fit(
        self,
        train_data: TimeSeriesDataFrame,
        val_data: Optional[TimeSeriesDataFrame] = None,
        time_limit: Optional[float] = None,
        **kwargs,
    ) -> None:
        """Fit the model on the available training data."""
        print("Entering the `_fit` method")

        # We lazily import other libraries inside the _fit method. This reduces the
        # import time for autogluon and ensures that even if one model has some problems
        # with dependencies, the training process won't crash
        import torch
        from neuralforecast import NeuralForecast
        from neuralforecast.losses.pytorch import MQLoss
        from neuralforecast.models import NHITS

        # Default hyperparameters for the model
        default_model_params = dict(
            loss=MQLoss(quantiles=self.quantile_levels),
            input_size=2 * self.prediction_length,
            scaler_type="standard",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            accelerator="cpu",
            # The model wrapper should handle any time series length - even time series
            # with 1 observation
            start_padding_enabled=True,
            # NeuralForecast requires that names of the past/future/static covariates are
            # passed as model arguments. AutoGluon models have access to this information
            # using the `metadata` attribute that is set automatically at model creation.
            #
            # Note that NeuralForecast does not support categorical covariates, so we
            # only use the real-valued covariates here. To use categorical features in
            # you wrapper, you need to either use techniques like one-hot-encoding, or
            # rely on models that natively handle categorical features.
            futr_exog_list=self.metadata.known_covariates_real,
            hist_exog_list=self.metadata.past_covariates_real,
            stat_exog_list=self.metadata.static_features_real,
        )
        if torch.cuda.is_available():
            default_model_params["accelerator"] = "gpu"
            default_model_params["devices"] = 1

        # It's important to ensure that the model respects the time_limit during `fit`.
        # Since NeuralForecast is based on PyTorch-Lightning, this can be easily enforced
        # using the `max_time` argument to `pl.Trainer`. For other model types such as
        # ARIMA implementing the time_limit logic may require a lot of work.
        if time_limit is not None:
            default_model_params["max_time"] = {"seconds": time_limit}

        # The method `_get_model_params()` returns the custom hyperparameters provided by
        # the user in `predictor.fit(..., hyperparameters={NHITSModel: {}})`. We override
        # the default hyperparameters defined above with the user hyperparameters.
        model_params = default_model_params | self._get_model_params()
        print(f"Hyperparameters:\n{pprint.pformat(model_params, sort_dicts=False)}")

        model = NHITS(h=self.prediction_length, **model_params)
        self.nf = NeuralForecast(models=[model], freq=self.freq)

        # Convert data into a format expected by the model. NeuralForecast expects time
        # series data in pandas.DataFrame format that is quite similar to AutoGluon, so
        # the transformation is very easy.
        #
        # Note that the `preprocess` method was already applied to train_data and val_data.
        train_df, static_df = self._to_neuralforecast_format(train_data)
        self.nf.fit(
            train_df,
            static_df=static_df,
            id_col="item_id",
            time_col="timestamp",
            target_col=self.target,
        )
        print("Exiting the `_fit` method")

    def _to_neuralforecast_format(self, data: TimeSeriesDataFrame) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Convert a TimeSeriesDataFrame to the format expected by NeuralForecast."""
        df = data.to_data_frame().reset_index()
        # Drop the categorical covariates to avoid NeuralForecast errors
        df = df.drop(columns=self.metadata.covariates_cat)
        static_df = data.static_features
        if len(self.metadata.static_features_real) > 0:
            static_df = static_df.reset_index()
            static_df = static_df.drop(columns=self.metadata.static_features_cat)
        return df, static_df

    def _predict(
        self,
        data: TimeSeriesDataFrame,
        known_covariates: Optional[TimeSeriesDataFrame] = None,
        **kwargs,
    ) -> TimeSeriesDataFrame:
        """Predict future target given the historic time series data and the future values of known_covariates."""
        print("Entering the `_predict` method")

        from neuralforecast.losses.pytorch import quantiles_to_outputs

        df, static_df = self._to_neuralforecast_format(data)
        if len(self.metadata.known_covariates_real) > 0:
            futr_df, _ = self._to_neuralforecast_format(known_covariates)
        else:
            futr_df = None

        with warning_filter():
            predictions = self.nf.predict(df, static_df=static_df, futr_df=futr_df)

        # predictions must be a TimeSeriesDataFrame with columns
        # ["mean"] + [str(q) for q in self.quantile_levels]
        model_name = str(self.nf.models[0])
        rename_columns = {
            f"{model_name}{suffix}": str(quantile)
            for quantile, suffix in zip(*quantiles_to_outputs(self.quantile_levels))
        }
        predictions = predictions.rename(columns=rename_columns)
        predictions["mean"] = predictions["0.5"]
        predictions = TimeSeriesDataFrame(predictions)
        return predictions
```

For convenience, here is an overview of the main constraints on the inputs and outputs of different methods.

- Input data received by `_fit` and `_predict` methods satisfies
    - the index is sorted by `(item_id, timestamp)`
    - timestamps of observations have a regular frequency corresponding to `self.freq`
    - column `self.target` contains the target values of the time series
    - target column might contain missing values represented by `NaN`
    - data may contain covariates (incl. static features) with schema described in `self.metadata`
        - real-valued covariates have dtype `float32`
        - categorical covariates have dtype `category`
        - covariates do not contain any missing values
    - static features, if present, are available as `data.static_features`
- Predictions returned by `_predict` must satisfy:
    - returns predictions as a `TimeSeriesDataFrame` object
    - predictions contain columns `["mean"] + [str(q) for q in self.quantile_levels]` containing the point and quantile forecasts, respectively
    - the index of predictions contains exactly `self.prediction_length` future time steps of each time series present in `data`
    - the frequency of the prediction timestamps matches `self.freq`
    - the index of predictions is sorted by `(item_id, timestamp)`
    - predictions contain no missing values represented by `NaN` and no gaps
- The runtime of `_fit` method should not exceed `time_limit` seconds, if `time_limit` is provided.
- None of the methods should modify the data in-place. If modifications are needed, create a copy of the data first.
- All methods should work even if some time series consist of all NaNs, or only have a single observation.

-----

We will now use this wrapper in two modes:
1. Standalone mode (outside the `TimeSeriesPredictor`).
    - This mode should be used for development and debugging. In this case, we need to take manually take care of preprocessing and model configuration.
2. Inside the `TimeSeriesPredictor`.
    - This mode makes it easy to combine & compare the custom model with other models available in AutoGluon. The main purpose of writing a custom model wrapper is to use it in this mode.

## Load and preprocess the data

First, we load the Grocery Sales dataset that we will use for development and evaluation.


```python
from autogluon.timeseries import TimeSeriesDataFrame

raw_data = TimeSeriesDataFrame.from_path(
    "https://autogluon.s3.amazonaws.com/datasets/timeseries/grocery_sales/test.csv",
    static_features_path="https://autogluon.s3.amazonaws.com/datasets/timeseries/grocery_sales/static.csv",
)
raw_data.head()
```


```python
raw_data.static_features.head()
```


```python
print("Types of the columns in raw data:")
print(raw_data.dtypes)
print("\nTypes of the columns in raw static features:")
print(raw_data.static_features.dtypes)

print("\nNumber of missing values per column:")
print(raw_data.isna().sum())
```

Define the forecasting task


```python
prediction_length = 7  # number of future steps to predict
target = "unit_sales"  # target column
known_covariates_names = ["promotion_email", "promotion_homepage"]  # covariates known in the future
```

Before we use the model in standalone mode, we need to apply the general AutoGluon preprocessing to the data.

The `TimeSeriesFeatureGenerator` captures preprocessing steps like normalizing the data types and imputing the missing values in the covariates.


```python
from autogluon.timeseries.utils.features import TimeSeriesFeatureGenerator

feature_generator = TimeSeriesFeatureGenerator(target=target, known_covariates_names=known_covariates_names)
data = feature_generator.fit_transform(raw_data)
```


```python
print("Types of the columns in preprocessed data:")
print(data.dtypes)
print("\nTypes of the columns in preprocessed static features:")
print(data.static_features.dtypes)

print("\nNumber of missing values per column:")
print(data.isna().sum())
```

## Using the custom model in standalone mode
Using the model in standalone mode is useful for debugging our implementation. Once we make sure that all methods work as expected, we will use the model inside the `TimeSeriesPredictor`.

### Training
We are now ready to train the custom model on the preprocessed data.

When using the model in standalone mode, we need to manually configure its parameters.


```python
model = NHITSModel(
    prediction_length=prediction_length,
    target=target,
    metadata=feature_generator.covariate_metadata,
    freq=data.freq,
    quantile_levels=[0.1, 0.5, 0.9],
)
model.fit(train_data=data, time_limit=20)
```

### Predicting and scoring


```python
past_data, known_covariates = data.get_model_inputs_for_scoring(
    prediction_length=prediction_length,
    known_covariates_names=known_covariates_names,
)
predictions = model.predict(past_data, known_covariates)
predictions.head()
```


```python
model.score(data)
```

## Using the custom model inside the `TimeSeriesPredictor`
After we made sure that our custom model works in standalone mode, we can pass it to the TimeSeriesPredictor alongside other models.


```python
from autogluon.timeseries import TimeSeriesPredictor

train_data, test_data = raw_data.train_test_split(prediction_length)

predictor = TimeSeriesPredictor(
    prediction_length=prediction_length,
    target=target,
    known_covariates_names=known_covariates_names,
)

predictor.fit(
    train_data,
    hyperparameters={
        "Naive": {},
        "Chronos": {"model_path": "bolt_small"},
        "ETS": {},
        NHITSModel: {},
    },
    time_limit=120,
)
```

Note that when we use the custom model inside the predictor, we don't need to worry about:
- manually configuring the model (setting `freq`, `prediction_length`)
- preprocessing the data using `TimeSeriesFeatureGenerator`
- setting the time limits

The `TimeSeriesPredictor` automatically takes care of all above aspects.

We can also easily compare our custom model with other model trained by the predictor.


```python
predictor.leaderboard(test_data)
```

We can also take advantage of other predictor functionality such as `feature_importance`.


```python
predictor.feature_importance(test_data, model="NHITS")
```

As expected, features `product_category` and `product_subcategory` have zero importance because our implementation ignores categorical features.

-----

Here is how we can train multiple versions of the custom model with different hyperparameter configurations


```python
predictor = TimeSeriesPredictor(
    prediction_length=prediction_length,
    target=target,
    known_covariates_names=known_covariates_names,
)
predictor.fit(
    train_data,
    hyperparameters={
        NHITSModel: [
            {},  # default hyperparameters
            {"input_size": 20},  # custom input_size
            {"scaler_type": "robust"},  # custom scaler_type
        ]
    },
    time_limit=60,
)
```


```python
predictor.leaderboard(test_data)
```

## Wrapping up

That's all it takes to add a custom forecasting model to AutoGluon. If you create a custom model, consider [submitting a PR](https://github.com/autogluon/autogluon/pulls) so that we can add it officially to AutoGluon!

For more tutorials, refer to [Forecasting Time Series - Quick Start](../forecasting-quick-start.ipynb) and [Forecasting Time Series - In Depth](../forecasting-indepth.ipynb).
