# Condensed: ```python

Summary: This tutorial demonstrates implementing the NHITS neural forecasting model in AutoGluon's time series framework. It covers creating a custom model class that handles preprocessing (filling missing values), configuring hyperparameters (with GPU support), converting data formats, and generating predictions with quantile levels. The implementation supports real-valued covariates (past, known, and static) while handling limitations like NaN values. The tutorial shows how to use the model both standalone and integrated with TimeSeriesPredictor for comparison against other models, hyperparameter tuning, and feature importance analysis. This knowledge helps with implementing custom neural forecasting models in AutoGluon's ecosystem.

*This is a condensed version that preserves essential implementation details and context.*

# NHITS Model Implementation for AutoGluon

## Core Implementation

```python
class NHITSModel(AbstractTimeSeriesModel):
    """AutoGluon-compatible wrapper for the NHITS model from NeuralForecast."""
    _supports_known_covariates: bool = True
    _supports_past_covariates: bool = True
    _supports_static_features: bool = True

    def preprocess(self, data: TimeSeriesDataFrame, known_covariates: Optional[TimeSeriesDataFrame] = None, 
                  is_train: bool = False, **kwargs) -> Tuple[TimeSeriesDataFrame, Optional[TimeSeriesDataFrame]]:
        # Handle missing values (NeuralForecast can't process NaNs)
        data = data.fill_missing_values()
        data = data.fill_missing_values(method="constant", value=0.0)
        return data, known_covariates
```

## Default Hyperparameters

```python
def _get_default_hyperparameters(self) -> dict:
    import torch
    from neuralforecast.losses.pytorch import MQLoss

    default_hyperparameters = dict(
        loss=MQLoss(quantiles=self.quantile_levels),
        input_size=2 * self.prediction_length,
        scaler_type="standard",
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        accelerator="cpu",
        start_padding_enabled=True,
        # Pass covariate information from metadata
        futr_exog_list=self.covariate_metadata.known_covariates_real,
        hist_exog_list=self.covariate_metadata.past_covariates_real,
        stat_exog_list=self.covariate_metadata.static_features_real,
    )

    if torch.cuda.is_available():
        default_hyperparameters["accelerator"] = "gpu"
        default_hyperparameters["devices"] = 1

    return default_hyperparameters
```

## Model Fitting

```python
def _fit(self, train_data: TimeSeriesDataFrame, val_data: Optional[TimeSeriesDataFrame] = None,
         time_limit: Optional[float] = None, **kwargs) -> None:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NHITS

    # Handle time limit if specified
    hyperparameter_overrides = {}
    if time_limit is not None:
        hyperparameter_overrides = {"max_time": {"seconds": time_limit}}

    # Get hyperparameters with overrides
    model_params = self.get_hyperparameters() | hyperparameter_overrides

    # Initialize and fit model
    model = NHITS(h=self.prediction_length, **model_params)
    self.nf = NeuralForecast(models=[model], freq=self.freq)

    # Convert data to NeuralForecast format
    train_df, static_df = self._to_neuralforecast_format(train_data)
    self.nf.fit(
        train_df,
        static_df=static_df,
        id_col="item_id",
        time_col="timestamp",
        target_col=self.target,
    )
```

## Data Format Conversion

```python
def _to_neuralforecast_format(self, data: TimeSeriesDataFrame) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Convert a TimeSeriesDataFrame to the format expected by NeuralForecast."""
    df = data.to_data_frame().reset_index()
    # Drop categorical covariates (not supported by NeuralForecast)
    df = df.drop(columns=self.covariate_metadata.covariates_cat)
    static_df = data.static_features
    if len(self.covariate_metadata.static_features_real) > 0:
        static_df = static_df.reset_index()
        static_df = static_df.drop(columns=self.covariate_metadata.static_features_cat)
    return df, static_df
```

## Prediction

```python
def _predict(self, data: TimeSeriesDataFrame, known_covariates: Optional[TimeSeriesDataFrame] = None, **kwargs) -> TimeSeriesDataFrame:
    from neuralforecast.losses.pytorch import quantiles_to_outputs

    df, static_df = self._to_neuralforecast_format(data)
    futr_df = None
    if len(self.covariate_metadata.known_covariates_real) > 0:
        futr_df, _ = self._to_neuralforecast_format(known_covariates)

    predictions = self.nf.predict(df, static_df=static_df, futr_df=futr_df)

    # Format predictions as TimeSeriesDataFrame with required columns
    model_name = str(self.nf.models[0])
    rename_columns = {
        f"{model_name}{suffix}": str(quantile)
        for quantile, suffix in zip(*quantiles_to_outputs(self.quantile_levels))
    }
    predictions = predictions.rename(columns=rename_columns)
    predictions["mean"] = predictions["0.5"]
    return TimeSeriesDataFrame(predictions)
```

## Important Notes
- NeuralForecast cannot handle NaN values, requiring preprocessing
- Only real-valued covariates are supported (categorical features are dropped)
- GPU acceleration is used when available
- Time limits are respected through PyTorch-Lightning's `max_time` parameter

# Custom Model Implementation in AutoGluon Time Series

## Installation

```python
!pip install uv
!uv pip install -q autogluon.timeseries --system
!uv pip uninstall -q torchaudio torchvision torchtext --system
!pip install -q neuralforecast==2.0
```

## Data Preparation

```python
from autogluon.timeseries import TimeSeriesDataFrame

raw_data = TimeSeriesDataFrame.from_path(
    "https://autogluon.s3.amazonaws.com/datasets/timeseries/grocery_sales/test.csv",
    static_features_path="https://autogluon.s3.amazonaws.com/datasets/timeseries/grocery_sales/static.csv",
)

# Define forecasting task parameters
prediction_length = 7  # number of future steps to predict
target = "unit_sales"  # target column
known_covariates_names = ["promotion_email", "promotion_homepage"]  # covariates known in the future

# Preprocess data
from autogluon.timeseries.utils.features import TimeSeriesFeatureGenerator
feature_generator = TimeSeriesFeatureGenerator(target=target, known_covariates_names=known_covariates_names)
data = feature_generator.fit_transform(raw_data)
```

## Using the Custom Model in Standalone Mode

```python
# Training
model = NHITSModel(
    prediction_length=prediction_length,
    target=target,
    covariate_metadata=feature_generator.covariate_metadata,
    freq=data.freq,
    quantile_levels=[0.1, 0.5, 0.9],
)
model.fit(train_data=data, time_limit=20)

# Predicting and scoring
past_data, known_covariates = data.get_model_inputs_for_scoring(
    prediction_length=prediction_length,
    known_covariates_names=known_covariates_names,
)
predictions = model.predict(past_data, known_covariates)
model.score(data)
```

## Using the Custom Model with TimeSeriesPredictor

```python
from autogluon.timeseries import TimeSeriesPredictor

train_data, test_data = raw_data.train_test_split(prediction_length)

predictor = TimeSeriesPredictor(
    prediction_length=prediction_length,
    target=target,
    known_covariates_names=known_covariates_names,
)

# Train with multiple models including our custom model
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

# Evaluate models
predictor.leaderboard(test_data)

# Feature importance
predictor.feature_importance(test_data, model="NHITS")
```

## Training with Multiple Hyperparameter Configurations

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

predictor.leaderboard(test_data)
```

## Key Points

1. Custom models must subclass `AbstractTimeSeriesModel` and implement `_fit` and `_predict` methods
2. For custom preprocessing logic, implement the `preprocess` method
3. When using with TimeSeriesPredictor, you don't need to manually configure:
   - Model parameters like `freq` and `prediction_length`
   - Data preprocessing
   - Time limits
4. The predictor allows easy comparison with other models and provides additional functionality like feature importance

If you create a custom model, consider submitting a PR to add it officially to AutoGluon.