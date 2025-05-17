# Condensed: Forecasting with Chronos

Summary: This tutorial covers implementing time series forecasting using AutoGluon's Chronos models, specifically focusing on the faster Chronos-Bolt variants. It demonstrates how to perform zero-shot forecasting and model fine-tuning, incorporate covariates using regression models, and compare model performances. Key implementation techniques include basic model setup, covariate integration through tabular regressors, and fine-tuning configurations with customizable learning rates and steps. The tutorial helps with tasks like time series prediction, model optimization, and handling exogenous variables. Notable features include support for both CPU and GPU execution, various model sizes (tiny to large), automated model selection through presets, and visualization capabilities. It emphasizes best practices for model selection, hardware requirements, and performance optimization.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on key implementation details and concepts:

# Forecasting with Chronos in AutoGluon

## Key Points
- Chronos models are pretrained on large collections of time series data
- New Chronos-Bolt⚡️ models are up to 250x faster than original Chronos
- Supports both zero-shot forecasting and fine-tuning

## Installation
```python
!pip install autogluon.timeseries
```

## Model Types & Presets

### Chronos-Bolt Models (Recommended)
- `bolt_tiny`, `bolt_mini`, `bolt_small`, `bolt_base`
- Can run on both CPU and GPU

### Original Chronos Models
- `chronos_tiny`, `chronos_mini`, `chronos_small`, `chronos_base`, `chronos_large`
- Models `small` and above require GPU

## Implementation Details

### Basic Usage
```python
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

# Load data
data = TimeSeriesDataFrame.from_path("path_to_data.csv")

# Split data
prediction_length = 48
train_data, test_data = data.train_test_split(prediction_length)

# Create and fit predictor
predictor = TimeSeriesPredictor(prediction_length=prediction_length).fit(
    train_data, 
    presets="bolt_small"
)

# Generate predictions
predictions = predictor.predict(train_data)
```

## Important Notes
1. Chronos models don't actually "fit" to data - computation happens during inference
2. Prediction computation scales linearly with number of time series
3. The `fit` method primarily:
   - Infers time series frequency
   - Saves predictor state
   - Handles basic setup tasks

## Best Practices
- Use Chronos-Bolt models for better performance
- For combining with other models, use presets:
  - `medium_quality`
  - `high_quality`
  - `best_quality`
- Consider GPU requirements when selecting model size

## Visualization
```python
predictor.plot(
    data=data,
    predictions=predictions,
    item_ids=data.item_ids[:2],
    max_history_length=200
)
```

This condensed version maintains all critical implementation details while removing redundant explanations and focusing on practical usage.

Here's the condensed tutorial section focusing on key implementation details and concepts:

# Incorporating Covariates with Chronos-Bolt

## Key Concepts
- Chronos is typically univariate but can incorporate exogenous variables through covariate regressors
- Covariate regressors are tabular models that predict target values using known covariates
- The process: regressor predictions are subtracted from target, then univariate model forecasts residuals

## Implementation Details

### 1. Data Setup
```python
data = TimeSeriesDataFrame.from_path("path_to_grocery_sales_data")
prediction_length = 8
train_data, test_data = data.train_test_split(prediction_length=prediction_length)
```

### 2. Predictor Configuration
```python
predictor = TimeSeriesPredictor(
    prediction_length=prediction_length,
    target="unit_sales",
    known_covariates_names=["scaled_price", "promotion_email", "promotion_homepage"],
).fit(
    train_data,
    hyperparameters={
        "Chronos": [
            # Zero-shot model without covariates
            {
                "model_path": "bolt_small",
                "ag_args": {"name_suffix": "ZeroShot"},
            },
            # Chronos-Bolt with CatBoost covariate regressor
            {
                "model_path": "bolt_small",
                "covariate_regressor": "CAT",
                "target_scaler": "standard",
                "ag_args": {"name_suffix": "WithRegressor"},
            },
        ],
    },
    enable_ensemble=False,
    time_limit=60,
)
```

## Important Configurations
1. `known_covariates_names`: List of covariate columns to use
2. `covariate_regressor`: Specifies the regression model (e.g., "CAT" for CatBoost)
3. `target_scaler`: Recommended when using covariate regressors
4. `model_path`: Specifies the Chronos-Bolt model size

## Best Practices
- Always apply target scaling when using covariate regressors
- Compare models with and without covariates to validate improvement
- Use meaningful name suffixes to distinguish models in the leaderboard

## Performance Note
Models using covariates typically achieve better forecast accuracy compared to pure univariate approaches, as demonstrated in the example where the model with covariates outperformed the zero-shot model.

Here's the condensed version focusing on the key implementation details and insights:

# Model Comparison and Best Practices

## Performance Comparison
```python
# Sample results table showing model performance
| model                           | score_test | score_val | pred_time_test | pred_time_val | fit_time_marginal |
|--------------------------------|------------|-----------|----------------|---------------|-------------------|
| ChronosWithRegressor[bolt_small]| -0.268969 | -0.358048 | 0.881176      | 0.916053     | 1.004376         |
| ChronosZeroShot[bolt_small]    | -0.318562 | -0.452296 | 0.859930      | 0.844927     | 0.019435         |
```

## Key Implementation Notes

1. **Model Selection**
   - Covariates may not always improve model performance
   - Zero-shot model might achieve better accuracy in some cases
   - Always evaluate multiple models on held-out data

2. **Best Practices**
   - Use AutoGluon's preset configurations:
     - `"high_quality"`
     - `"best_quality"`
   - These presets automatically try multiple models and select the best performer

3. **Performance Metrics**
   - Consider both prediction accuracy and computational costs
   - Monitor fit time and prediction time alongside accuracy metrics

This section emphasizes the importance of empirical model selection and the use of AutoGluon's automated model selection capabilities.

Here's the condensed version focusing on key implementation details and practices:

# Fine-tuning Chronos Models

## Core Implementation

### Basic Fine-tuning Setup
```python
predictor = TimeSeriesPredictor(prediction_length=prediction_length).fit(
    train_data=train_data,
    hyperparameters={
        "Chronos": [
            # Zero-shot configuration
            {"model_path": "bolt_small", "ag_args": {"name_suffix": "ZeroShot"}},
            # Fine-tuned configuration
            {"model_path": "bolt_small", "fine_tune": True, "ag_args": {"name_suffix": "FineTuned"}},
        ]
    },
    time_limit=60,
    enable_ensemble=False,
)
```

### Custom Fine-tuning Parameters
```python
predictor.fit(
    ...,
    hyperparameters={
        "Chronos": {
            "fine_tune": True,
            "fine_tune_lr": 1e-4,
            "fine_tune_steps": 2000
        }
    }
)
```

## Key Points

1. **Model Variants**:
   - Zero-shot: Uses pretrained model directly
   - Fine-tuned: Adapts model to specific dataset

2. **Performance Evaluation**:
   - Use `predictor.leaderboard(test_data)` to compare model variants
   - Scores are reported in "higher is better" format (error metrics are multiplied by -1)

3. **Hardware Requirements**:
   - Recommended: AWS g5.2xlarge or p3.2xlarge
   - GPU: 16GB+ memory
   - RAM: 32GB+ recommended
   - CPU execution possible but slower

## Best Practices

1. **Model Selection**:
   - Fine-tuned models typically achieve better accuracy than zero-shot
   - Compare performance using leaderboard functionality

2. **Resource Considerations**:
   - GPU recommended for larger models and fine-tuning
   - CPU mode available but with longer runtime

3. **Support Channels**:
   - Discord server: discord.gg/wjUmjqAc2N
   - GitHub: github.com/autogluon/autogluon
   - Chronos GitHub: github.com/amazon-science/chronos-forecasting/discussions

For detailed fine-tuning options, refer to the Chronos documentation in the Forecasting Model Zoo.