# Condensed: ```python

Summary: This tutorial covers AutoGluon's TabularPredictor for automated machine learning on tabular data, including classification and regression tasks. It demonstrates minimal 2-line training code, automatic problem type detection, feature engineering, model ensembling, and hyperparameter tuning. Key implementations include data loading with TabularDataset (CSV/Parquet support), prediction methods (predict/predict_proba), model evaluation (evaluate/leaderboard), feature importance analysis, and model persistence. Critical configurations covered: `presets` parameter ('extreme' for GPU, 'best' for CPU production), `eval_metric` specification (f1, roc_auc, RMSE, custom metrics), `time_limit` for training duration, and best practices like avoiding manual preprocessing, using raw data, and not specifying hyperparameters manually. The tutorial helps with building production ML pipelines, optimizing model performance, selecting appropriate presets based on hardware/requirements, and understanding AutoGluon's automatic feature handling and ensemble strategies.

*This is a condensed version that preserves essential implementation details and context.*

# AutoGluon Tabular Quick Start

## Installation & Setup

```python
!pip install autogluon.tabular[all]
from autogluon.tabular import TabularDataset, TabularPredictor
```

## Data Loading

Load data using `TabularDataset` (wrapper around Pandas DataFrame):

```python
train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
# Also works with local files and parquet format
subsample_size = 1000
train_data = train_data.sample(n=subsample_size, random_state=0)
```

**Best Practice:** Use raw data without preprocessing. Avoid manual missing value imputation or one-hot-encoding—AutoGluon handles these automatically.

## Training

Minimal training code:

```python
label = 'class'
predictor = TabularPredictor(label=label).fit(train_data)
```

## Prediction

```python
test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')

# Class predictions
y_pred = predictor.predict(test_data)

# Prediction probabilities
y_pred_proba = predictor.predict_proba(test_data)
```

## Evaluation

```python
# Overall evaluation
predictor.evaluate(test_data)

# Per-model leaderboard
predictor.leaderboard(test_data)
```

## Saving & Loading

```python
# Get save path
predictor.path

# Load predictor
predictor = TabularPredictor.load(predictor.path)
```

```{warning}
`TabularPredictor.load()` uses `pickle`, which is insecure. **Only load data from trusted sources** as malicious pickle data can execute arbitrary code.
```

## Minimal Production Code

```python
from autogluon.tabular import TabularPredictor
predictor = TabularPredictor(label="your_label_name").fit(train_data="train_data.csv")
```

**Note:** For maximum performance, specify `presets` parameter in `fit()` and `eval_metric` in `TabularPredictor()` (covered in subsequent sections).

# Understanding fit() Process

## Automatic Inference

AutoGluon automatically handles:
- **Problem type detection** (binary/multiclass classification, regression)
- **Feature type inference** (continuous vs. categorical)
- **Missing data and feature rescaling**
- **Train/validation split** (if validation data not provided)

```python
# View inferred properties
print("AutoGluon infers problem type is: ", predictor.problem_type)
print("AutoGluon identified the following types of features:")
print(predictor.feature_metadata)
```

## Model Training Strategy

- Trains **multiple model types** (neural networks, tree ensembles, etc.)
- **Automatic hyperparameter tuning** via iterative testing
- **Parallelization** using Ray for faster training
- **Ensembling** of models for superior performance
- Control runtime with `time_limit` parameter in `fit()`

## Data Transformation

View AutoGluon's internal data representation:

```python
test_data_transform = predictor.transform_features(test_data)
```

Data becomes purely numeric after preprocessing (categorical features still treated as categorical downstream).

## Feature Importance

```python
predictor.feature_importance(test_data)
```

- `importance` column: estimated metric drop if feature removed
- **Negative values**: removing feature may improve results

## Model Selection

```python
# Best performing model (default for predict())
predictor.model_best

# Predict with specific model
predictor.predict(test_data, model='LightGBM')

# List all trained models
predictor.model_names()
```

**Note:** Default predictions use the weighted-ensemble (best validation performance). Specify custom `eval_metric` in `TabularPredictor()` if your application requires a different performance metric than the default (accuracy for binary classification).

# Presets & Performance Optimization

## Presets Overview

Specify via `presets` argument in `.fit()`:

| Preset | Quality | Use Case | Fit Time | Inference Time | Disk Usage |
|:-------|:--------|:---------|:---------|:---------------|:-----------|
| **extreme** | **Far better** than best (<100k samples) | Cutting-edge tabular foundation models (Nori, TabICLv2, TabDPT-Turbo). **Requires GPU**. All models free for commercial use. | 1x | 8x | 2x |
| **noncommercial** | **Far better** than best (<100k samples) | `extreme` + TabPFN-3. **Requires GPU**. TabPFN-3 needs Prior Labs license for commercial use. | 1x | 8x | 2x |
| **best** | SOTA, much better than high | **Preferred for serious usage**. Maximum accuracy. Used to win Kaggle competitions. | 16x+ | 32x+ | 16x+ |
| **high** | Better than good | Powerful + portable with fast inference. Large-scale batch inference. | 16x+ | 4x | 2x |
| **good** | Stronger than other AutoML | Highly portable, very fast inference. Billion-scale batch, sub-100ms online inference, edge devices. | 16x | 2x | 0.1x |
| **medium** | Competitive with top AutoML | **Default**. Initial prototyping, baseline establishment. | 1x | 1x | 1x |

Install foundation model dependencies: `pip install autogluon[tabarena]`

## Preset Selection Strategy

**With GPU:** Start with `extreme` (meta-learned from TabArena, free for commercial use)
**Without GPU:** Start with `best` for strong baseline

If training too slow: Use `medium` or subsample data during prototyping. Always hold out test data for validation.

## Maximizing Predictive Performance

**Critical:** Don't use default arguments for benchmarking or maximum accuracy!

```python
time_limit = 60  # Set to maximum time you can wait (seconds)
metric = 'roc_auc'  # Specify your evaluation metric
predictor = TabularPredictor(label, eval_metric=metric).fit(
    train_data, 
    time_limit=time_limit, 
    presets='best'
)
```

### Best Practices for Maximum Accuracy

1. **Use `presets='best'`** - Enables powerful stacking/bagging ensembles
2. **Specify `eval_metric`** - Match your application's metric:
   - Binary classification: `'f1'`, `'roc_auc'`
   - Classification: `'log_loss'`
   - Regression: `'mean_absolute_error'`, `'median_absolute_error'`
   - Custom metrics supported
3. **Include all data in `train_data`** - Don't provide `tuning_data` (AutoGluon splits intelligently)
4. **Don't specify `hyperparameter_tune_kwargs`** - Model ensembling > hyperparameter tuning for limited time budgets (only use for single model deployment)
5. **Don't specify `hyperparameters`** - Let AutoGluon adaptively select models
6. **Set `time_limit`** - Longer training = better performance

**Fast deployment alternative:** `presets=['good', 'optimize_for_deployment']`

# Regression & Advanced Topics

## Regression Tasks

AutoGluon automatically handles regression (predicting numeric columns):

```python
age_column = 'age'
predictor_age = TabularPredictor(label=age_column, path="agModels-predictAge").fit(
    train_data, 
    time_limit=30
)
```

```python
predictor_age.leaderboard(test_data)
```

**Key Points:**
- **Automatic problem detection** - No need to specify regression vs. classification
- **Default metric:** RMSE for regression
- **Custom metrics:** Set `eval_metric` parameter (e.g., `eval_metric='mean_absolute_error'`)
- **Metric sign convention:** Higher-is-worse metrics (like RMSE) displayed as negative values internally

**Custom Metrics:** Follow [Custom Metric Tutorial](advanced/tabular-custom-metric.ipynb)

## Supported Data Formats

- **Pandas DataFrames** (in-memory)
- **CSV files**
- **Parquet files**

**Important:** Data must be in a single table with:
- Rows = statistically independent observations
- Columns = features/variables/covariates

Join multiple tables before using AutoGluon.

## Further Resources

- **Advanced usage:** [In Depth Tutorial](tabular-indepth.ipynb)
- **Deployment optimization:** [Deployment Optimization Tutorial](advanced/tabular-deployment.ipynb)
- **Custom models:** [Custom Model Tutorial](advanced/tabular-custom-model.ipynb) and [Custom Model Advanced](advanced/tabular-custom-model-advanced.ipynb)
- **Full API:** [TabularPredictor documentation](../../api/autogluon.tabular.TabularPredictor.rst)