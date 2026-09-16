# Condensed: ```python

Summary: This tutorial demonstrates creating custom time series forecasting models for AutoGluon by wrapping NeuralForecast's NHITS model. It covers implementing `AbstractTimeSeriesModel` subclasses with required `_fit`, `_predict`, and `preprocess` methods, handling NaN values through forward/backward filling, configuring hyperparameters with covariate support (real-valued only, no categorical), enforcing time limits via PyTorch-Lightning, and converting between AutoGluon's TimeSeriesDataFrame and NeuralForecast formats. Key techniques include lazy imports, GPU acceleration detection, quantile prediction formatting, and using `TimeSeriesFeatureGenerator` for preprocessing. The tutorial shows both standalone model usage for debugging and integration with `TimeSeriesPredictor` for automated configuration, model comparison via leaderboards, feature importance analysis, and training multiple hyperparameter configurations simultaneously.

*This is a condensed version that preserves essential implementation details and context.*

# NHITS Model Implementation Guide

## Logger Configuration (Optional)
```python
# Disable PyTorch-Lightning verbose logging
for logger_name in [
    "lightning.pytorch.utilities.rank_zero",
    "pytorch_lightning.accelerators.cuda",
    "lightning_fabric.utilities.seed",
]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)
```

## Model Class Structure

```python
class NHITSModel(AbstractTimeSeriesModel):
    """AutoGluon-compatible wrapper for NeuralForecast NHITS model."""
    
    # Enable covariate support
    _supports_known_covariates: bool = True
    _supports_past_covariates: bool = True
    _supports_static_features: bool = True
```

## Key Methods

### 1. Preprocessing
**Critical**: NeuralForecast cannot handle NaN values.

```python
def preprocess(self, data: TimeSeriesDataFrame, known_covariates: Optional[TimeSeriesDataFrame] = None, 
               is_train: bool = False, **kwargs):
    # Forward/backward fill missing values
    data = data.fill_missing_values()
    # Fill remaining NaNs (all-missing series) with 0.0
    data = data.fill_missing_values(method="constant", value=0.0)
    return data, known_covariates
```

### 2. Default Hyperparameters

```python
def _get_default_hyperparameters(self) -> dict:
    from neuralforecast.losses.pytorch import MQLoss
    
    default_hyperparameters = dict(
        loss=MQLoss(quantiles=self.quantile_levels),
        input_size=2 * self.prediction_length,
        scaler_type="standard",
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        accelerator="cpu",
        start_padding_enabled=True,  # Handle series with 1 observation
        # Pass covariate names (real-valued only, no categorical)
        futr_exog_list=self.covariate_metadata.known_covariates_real,
        hist_exog_list=self.covariate_metadata.past_covariates_real,
        stat_exog_list=self.covariate_metadata.static_features_real,
    )
    
    # GPU acceleration if available
    if torch.cuda.is_available():
        default_hyperparameters["accelerator"] = "gpu"
        default_hyperparameters["devices"] = 1
    
    return default_hyperparameters
```

**Note**: NeuralForecast doesn't support categorical covariates—use only real-valued features or apply one-hot encoding.

### 3. Training (_fit)

```python
def _fit(self, train_data: TimeSeriesDataFrame, val_data: Optional[TimeSeriesDataFrame] = None,
         time_limit: Optional[float] = None, **kwargs):
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NHITS
    
    # Enforce time_limit using PyTorch-Lightning's max_time
    hyperparameter_overrides = {}
    if time_limit is not None:
        hyperparameter_overrides = {"max_time": {"seconds": time_limit}}
    
    # Merge default, user-provided, and runtime hyperparameters
    model_params = self.get_hyperparameters() | hyperparameter_overrides
    
    model = NHITS(h=self.prediction_length, **model_params)
    self.nf = NeuralForecast(models=[model], freq=self.freq)
    
    # Convert to NeuralForecast format (preprocess already applied)
    train_df, static_df = self._to_neuralforecast_format(train_data)
    self.nf.fit(train_df, static_df=static_df, id_col="item_id", 
                time_col="timestamp", target_col=self.target)
```

**Best Practice**: Lazy import libraries inside `_fit` to reduce import time and prevent dependency crashes.

### 4. Data Format Conversion

```python
def _to_neuralforecast_format(self, data: TimeSeriesDataFrame) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    df = data.to_data_frame().reset_index()
    # Drop categorical covariates to avoid errors
    df = df.drop(columns=self.covariate_metadata.covariates_cat)
    
    static_df = data.static_features
    if len(self.covariate_metadata.static_features_real) > 0:
        static_df = static_df.reset_index()
        static_df = static_df.drop(columns=self.covariate_metadata.static_features_cat)
    return df, static_df
```

### 5. Prediction (_predict)

```python
def _predict(self, data: TimeSeriesDataFrame, known_covariates: Optional[TimeSeriesDataFrame] = None, **kwargs):
    from neuralforecast.losses.pytorch import quantiles_to_outputs
    
    df, static_df = self._to_neuralforecast_format(data)
    
    # Prepare future covariates if available
    futr_df = None
    if len(self.covariate_metadata.known_covariates_real) > 0:
        futr_df, _ = self._to_neuralforecast_format(known_covariates)
    
    with warning_filter():
        predictions = self.nf.predict(df, static_df=static_df, futr_df=futr_df)
    
    # Rename columns to match AutoGluon format: ["mean"] + quantile levels
    model_name = str(self.nf.models[0])
    rename_columns = {
        f"{model_name}{suffix}": str(quantile)
        for quantile, suffix in zip(*quantiles_to_outputs(self.quantile_levels))
    }
    predictions = predictions.rename(columns=rename_columns)
    predictions["mean"] = predictions["0.5"]
    
    return TimeSeriesDataFrame(predictions)
```

**Output Format**: Must return TimeSeriesDataFrame with columns `["mean"]` + stringified quantile levels.

# Custom Model Integration & Usage Guide

## Installation

```python
# Fast installation with uv
!pip install uv
!uv pip install -q autogluon.timeseries --system
!uv pip uninstall -q torchaudio torchvision torchtext --system  # Fix Colab incompatibilities
!pip install -q neuralforecast==2.0
```

## Implementation Requirements

**Core Requirement**: Subclass [`AbstractTimeSeriesModel`](https://github.com/autogluon/autogluon/blob/master/timeseries/src/autogluon/timeseries/models/abstract/abstract_timeseries_model.py) and implement:
- `_fit`: Training logic
- `_predict`: Prediction logic
- `preprocess` (optional): Custom preprocessing (e.g., missing value handling)

## Data Preparation

### Load Data
```python
from autogluon.timeseries import TimeSeriesDataFrame

raw_data = TimeSeriesDataFrame.from_path(
    "https://autogluon.s3.amazonaws.com/datasets/timeseries/grocery_sales/test.csv",
    static_features_path="https://autogluon.s3.amazonaws.com/datasets/timeseries/grocery_sales/static.csv",
)
```

### Define Task Parameters
```python
prediction_length = 7
target = "unit_sales"
known_covariates_names = ["promotion_email", "promotion_homepage"]
```

### Apply AutoGluon Preprocessing
**Critical**: Use `TimeSeriesFeatureGenerator` for type normalization and covariate imputation before standalone model usage.

```python
from autogluon.timeseries.utils.features import TimeSeriesFeatureGenerator

feature_generator = TimeSeriesFeatureGenerator(
    target=target, 
    known_covariates_names=known_covariates_names
)
data = feature_generator.fit_transform(raw_data)
```

## Standalone Model Usage

### Training
```python
model = NHITSModel(
    prediction_length=prediction_length,
    target=target,
    covariate_metadata=feature_generator.covariate_metadata,
    freq=data.freq,
    quantile_levels=[0.1, 0.5, 0.9],
)
model.fit(train_data=data, time_limit=20)
```

### Prediction & Scoring
```python
past_data, known_covariates = data.get_model_inputs_for_scoring(
    prediction_length=prediction_length,
    known_covariates_names=known_covariates_names,
)
predictions = model.predict(past_data, known_covariates)
model.score(data)
```

**Use Case**: Standalone mode is ideal for debugging before integrating with `TimeSeriesPredictor`.

## Integration with TimeSeriesPredictor

### Basic Usage
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
        NHITSModel: {},  # Custom model
    },
    time_limit=120,
)
```

**Benefits**: Predictor automatically handles:
- Model configuration (`freq`, `prediction_length`)
- Data preprocessing via `TimeSeriesFeatureGenerator`
- Time limit enforcement

### Model Comparison & Analysis
```python
# Compare models
predictor.leaderboard(test_data)

# Feature importance
predictor.feature_importance(test_data, model="NHITS")
```

### Multiple Hyperparameter Configurations
```python
predictor.fit(
    train_data,
    hyperparameters={
        NHITSModel: [
            {},                              # Default hyperparameters
            {"input_size": 20},              # Custom input_size
            {"scaler_type": "robust"},       # Custom scaler_type
        ]
    },
    time_limit=60,
)
```

## Best Practices

1. **Development Workflow**: Test in standalone mode → integrate with predictor
2. **Preprocessing**: Use `TimeSeriesFeatureGenerator` for consistent data handling
3. **Contribution**: Consider [submitting a PR](https://github.com/autogluon/autogluon/pulls) for official inclusion

## Additional Resources
- [Forecasting Time Series - Quick Start](../forecasting-quick-start.ipynb)
- [Forecasting Time Series - In Depth](../forecasting-indepth.ipynb)