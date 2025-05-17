# Condensed: Adding a custom time series forecasting model

Summary: This tutorial demonstrates the implementation of a NHITS (Neural Hierarchical Interpolation for Time Series) model within AutoGluon's framework. It provides implementation details for model class setup, data preprocessing, model fitting with GPU support, and data format conversion. The tutorial helps with tasks like handling missing values, integrating deep learning models into AutoGluon, managing time limits, and supporting various feature types (known covariates, past covariates, static features). Key functionalities covered include GPU acceleration, quantile-based predictions, automatic preprocessing, and compatibility with AutoGluon's ensemble capabilities. The implementation knowledge is particularly valuable for developers looking to integrate custom neural forecasting models into the AutoGluon ecosystem.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on key implementation details:

# NHITS Model Implementation for AutoGluon

## Key Components

### Model Class Setup
```python
class NHITSModel(AbstractTimeSeriesModel):
    _supports_known_covariates = True
    _supports_past_covariates = True
    _supports_static_features = True
```

### Data Preprocessing
```python
def preprocess(self, data, known_covariates=None, is_train=False, **kwargs):
    # Handle missing values
    data = data.fill_missing_values()
    data = data.fill_missing_values(method="constant", value=0.0)
    return data, known_covariates
```

### Model Fitting
```python
def _fit(self, train_data, val_data=None, time_limit=None, **kwargs):
    # Default configuration
    default_model_params = {
        'loss': MQLoss(quantiles=self.quantile_levels),
        'input_size': 2 * self.prediction_length,
        'scaler_type': "standard",
        'start_padding_enabled': True,
        'futr_exog_list': self.metadata.known_covariates_real,
        'hist_exog_list': self.metadata.past_covariates_real,
        'stat_exog_list': self.metadata.static_features_real
    }
    
    # GPU support
    if torch.cuda.is_available():
        default_model_params.update({
            "accelerator": "gpu",
            "devices": 1
        })
    
    # Time limit handling
    if time_limit:
        default_model_params["max_time"] = {"seconds": time_limit}
    
    # Initialize and fit model
    model = NHITS(h=self.prediction_length, **model_params)
    self.nf = NeuralForecast(models=[model], freq=self.freq)
```

### Data Format Conversion
```python
def _to_neuralforecast_format(self, data):
    df = data.to_data_frame().reset_index()
    df = df.drop(columns=self.metadata.covariates_cat)
    static_df = data.static_features
    if len(self.metadata.static_features_real) > 0:
        static_df = static_df.reset_index()
        static_df = static_df.drop(columns=self.metadata.static_features_cat)
    return df, static_df
```

## Important Notes

1. **Missing Values**: NeuralForecast cannot handle NaN values - preprocessing is required
2. **Categorical Features**: NeuralForecast doesn't support categorical covariates natively
3. **Time Limits**: Implements time limit constraints using PyTorch-Lightning's `max_time`
4. **GPU Support**: Automatically enables GPU acceleration when available

## Best Practices

1. Lazy import dependencies inside `_fit` method to reduce import time
2. Handle missing values through preprocessing
3. Implement proper time limit handling
4. Convert data formats appropriately for model compatibility
5. Drop categorical covariates to avoid errors

Here's the condensed tutorial focusing on key implementation details and practices:

# Adding Custom Time Series Forecasting Models in AutoGluon

## Key Implementation Requirements

1. Create a subclass of `AbstractTimeSeriesModel`
2. Implement required methods:
   - `_fit`
   - `_predict`
   - `preprocess` (if custom preprocessing needed)

## Critical Code Components

```python
class NHITSModel(AbstractTimeSeriesModel):
    def __init__(
        self,
        prediction_length: int,
        target: str,
        metadata: dict,
        freq: str,
        quantile_levels: Optional[List[float]] = None,
        **kwargs,
    ):
        super().__init__(
            prediction_length=prediction_length,
            target=target,
            metadata=metadata,
            freq=freq,
            **kwargs,
        )
        self.quantile_levels = quantile_levels or [0.1, 0.5, 0.9]
```

## Important Configurations

1. Data Preprocessing:
```python
feature_generator = TimeSeriesFeatureGenerator(
    target=target, 
    known_covariates_names=known_covariates_names
)
data = feature_generator.fit_transform(raw_data)
```

2. Model Usage:
```python
# Standalone Mode
model = NHITSModel(
    prediction_length=prediction_length,
    target=target,
    metadata=feature_generator.covariate_metadata,
    freq=data.freq,
    quantile_levels=[0.1, 0.5, 0.9],
)

# Within TimeSeriesPredictor
predictor = TimeSeriesPredictor(
    prediction_length=prediction_length,
    target=target,
    known_covariates_names=known_covariates_names,
)
```

## Best Practices

1. Test custom model in standalone mode before integration
2. Use TimeSeriesPredictor for automatic:
   - Model configuration
   - Data preprocessing
   - Time limit handling

3. Multiple model configurations:
```python
predictor.fit(
    train_data,
    hyperparameters={
        NHITSModel: [
            {},  # default
            {"input_size": 20},
            {"scaler_type": "robust"},
        ]
    },
    time_limit=60,
)
```

## ⚠️ Important Warnings

1. Custom model implementations rely on private AutoGluon API
2. May require updates when upgrading AutoGluon versions
3. Ensure proper handling of data types and missing values

## Integration Features

- Compatible with AutoGluon's model comparison tools
- Supports feature importance analysis
- Can be ensembled with other models
- Supports hyperparameter tuning

This implementation allows custom models to be trained, tuned, and ensembled alongside AutoGluon's default forecasting models.