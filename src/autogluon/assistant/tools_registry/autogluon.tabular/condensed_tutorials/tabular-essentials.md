# Condensed: ```python

Summary: This tutorial covers AutoGluon's TabularPredictor for automated machine learning on tabular data. It demonstrates implementation of quick model training with minimal code, automated handling of data preprocessing, and model deployment. Key functionalities include: loading tabular data, training multiple models simultaneously, making predictions, evaluating performance, and saving/loading models. The tutorial explains different performance presets (from "medium" to "extreme"), feature importance analysis, and optimization strategies for classification and regression tasks. AutoGluon automatically handles missing values, feature engineering, and model selection, making it valuable for rapid prototyping and production-quality predictive modeling with just a few lines of code.

*This is a condensed version that preserves essential implementation details and context.*

# AutoGluon TabularPredictor Quick Start

## Setup and Installation

```python
!pip install autogluon.tabular[all]
from autogluon.tabular import TabularDataset, TabularPredictor
```

## Loading Data

```python
# Load data (TabularDataset is essentially a Pandas DataFrame)
train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
train_data = train_data.sample(n=500, random_state=0)  # Subsample for demo

# Identify the target column
label = 'class'
print(f"Unique classes: {list(train_data[label].unique())}")
```

**Important:** AutoGluon works with raw data - avoid preprocessing like missing value imputation or one-hot-encoding as AutoGluon handles these automatically.

## Training

```python
# Initialize and fit in one line
predictor = TabularPredictor(label=label).fit(train_data)
```

## Prediction

```python
# Load test data
test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')

# Make predictions
y_pred = predictor.predict(test_data)
y_pred_proba = predictor.predict_proba(test_data)  # For probability predictions
```

## Evaluation

```python
# Evaluate overall performance
predictor.evaluate(test_data)

# Evaluate individual models
predictor.leaderboard(test_data)
```

## Saving and Loading

```python
# The predictor is automatically saved during training
predictor_path = predictor.path

# Load the predictor in a new session
predictor = TabularPredictor.load(predictor_path)
```

⚠️ **WARNING:** `TabularPredictor.load()` uses the `pickle` module which can be insecure. Only load data from trusted sources.

## Minimal Usage

For your own datasets, you can use just two lines of code:

```python
from autogluon.tabular import TabularPredictor
predictor = TabularPredictor(label='your_target_column').fit(train_data='your_data.csv')
```

Note: This simple call is intended for prototyping. For better performance, specify the `presets` parameter to `fit()` and the `eval_metric` parameter to `TabularPredictor()`.

# AutoGluon Tabular Fit Process and Presets

## Understanding the `fit()` Process

AutoGluon automatically handles binary classification problems, inferring feature types and addressing common issues like missing data and feature scaling:

```python
print("AutoGluon infers problem type is: ", predictor.problem_type)
print("AutoGluon identified the following types of features:")
print(predictor.feature_metadata)
```

During training:
- AutoGluon automatically splits data into training/validation sets when not specified
- Trains multiple models of various types (neural networks, tree ensembles, etc.)
- Automatically tunes hyperparameters to maximize validation performance
- Parallelizes training across multiple threads using Ray
- Creates model ensembles to improve predictive performance

## Data Transformation and Feature Importance

View the transformed data in AutoGluon's internal representation:

```python
test_data_transform = predictor.transform_features(test_data)
test_data_transform.head()
```

Analyze feature importance to understand model decisions:

```python
predictor.feature_importance(test_data)
```

The `importance` column estimates how much the evaluation metric would drop if the feature were removed.

## Working with Models

By default, AutoGluon predicts with the best-performing model:

```python
predictor.model_best  # View the best model
predictor.model_names()  # List all trained models
```

Specify a particular model for predictions:
```python
predictor.predict(test_data, model='LightGBM')
```

## Presets for Different Use Cases

| Preset | Model Quality | Use Cases | Fit Time | Inference Time | Disk Usage |
|--------|---------------|-----------|----------|----------------|------------|
| extreme | Far better than best on datasets <30K samples | Cutting edge with tabular foundation models (TabPFNv2, TabICL, Mitra, TabM). Requires GPU. | 4x+ | 32x+ | 8x+ |
| best | State-of-the-art | For serious usage, competition-winning quality | 16x+ | 32x+ | 16x+ |
| high | Better than good | Powerful, portable solution with fast inference | 16x+ | 4x | 2x |
| good | Stronger than other AutoML frameworks | Fast inference for large-scale/edge deployment | 16x | 2x | 0.1x |
| medium | Competitive with top AutoML frameworks | Initial prototyping, baseline performance | 1x | 1x | 1x |

**Recommended workflow:**
1. Start with `medium` preset for initial prototyping
2. Move to `best` preset with at least 16x the time limit for production-quality models
3. Try `extreme` preset on GPU for small datasets (<30K samples)
4. Consider `high` or `good` presets if inference speed or model size are critical

For GPU users, install additional dependencies with: `pip install autogluon[tabarena]`

# Maximizing Predictive Performance

For best predictive accuracy with AutoGluon, use this approach:

```python
time_limit = 60  # set to maximum time you're willing to wait (in seconds)
metric = 'roc_auc'  # specify your evaluation metric
predictor = TabularPredictor(label, eval_metric=metric).fit(train_data, time_limit=time_limit, presets='best')
```

```python
predictor.leaderboard(test_data)
```

## Key Strategies for Maximum Accuracy

- Use `presets='best'` to enable powerful model ensembles with stacking/bagging
  - Default is `'medium'` (less accurate but faster)
  - For faster deployment with lower accuracy: `presets=['good', 'optimize_for_deployment']`

- Specify `eval_metric` based on your evaluation needs:
  - Classification: `'f1'`, `'roc_auc'`, `'log_loss'`
  - Regression: `'mean_absolute_error'`, `'median_absolute_error'`
  - Custom metrics are supported (see [Adding a custom metric](advanced/tabular-custom-metric.ipynb))

- Include all data in `train_data` without providing `tuning_data`

- Avoid specifying `hyperparameter_tune_kwargs` (model ensembling is often superior)

- Don't specify `hyperparameters` (let AutoGluon select models adaptively)

- Set `time_limit` to the maximum time you can allow (longer time = better performance)

## Regression Example

To predict a numeric column like `age`:

```python
predictor_age = TabularPredictor(label=age_column, path="agModels-predictAge").fit(train_data, time_limit=60)
predictor_age.evaluate(test_data)
predictor_age.leaderboard(test_data)
```

AutoGluon automatically:
- Detects regression tasks from the data
- Reports appropriate metrics (RMSE by default)
- Flips signs for metrics where lower is better (internally assumes higher is better)

## Data Format Support
- Pandas DataFrames
- CSV files
- Parquet files

## Advanced Usage
- For more advanced examples: [In Depth Tutorial](tabular-indepth.ipynb)
- For deployment optimization: [Deployment Optimization Tutorial](advanced/tabular-deployment.ipynb)
- For custom models: [Custom Model](advanced/tabular-custom-model.ipynb) and [Custom Model Advanced](advanced/tabular-custom-model-advanced.ipynb) tutorials