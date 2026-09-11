# Condensed: ```python

Summary: This tutorial teaches how to create custom machine learning models in AutoGluon by inheriting from AbstractModel and implementing four key methods: `_preprocess` (data transformation with LabelEncoderFeatureGenerator), `_fit` (model training with problem-type detection), `_set_default_params` (hyperparameter defaults), and `_get_default_auxiliary_params` (supported data types). It covers the complete workflow from standalone model training with manual label/feature cleaning using LabelCleaner and AutoMLPipelineFeatureGenerator, to integration with TabularPredictor for automated preprocessing, bagging ensembles with BaggedEnsembleModel for performance improvement, hyperparameter tuning using search spaces (space.Int, space.Real, space.Categorical), and combining custom models with AutoGluon's default models using get_hyperparameter_config. Key techniques include importing dependencies within methods for modularity, handling categorical features and missing values, model persistence, and creating multi-layer stack ensembles with the 'best_quality' preset.

*This is a condensed version that preserves essential implementation details and context.*

# Custom Model Implementation in AutoGluon

## Installation & Setup

```python
!pip install autogluon.tabular[all]
```

## Custom Model Implementation

### Core Model Structure

```python
import numpy as np
import pandas as pd
from autogluon.core.models import AbstractModel
from autogluon.features.generators import LabelEncoderFeatureGenerator

class CustomRandomForestModel(AbstractModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator = None
```

### Data Preprocessing

**Key Implementation**: The `_preprocess` method transforms input data to model-compatible format.

```python
def _preprocess(self, X: pd.DataFrame, is_train=False, **kwargs) -> np.ndarray:
    X = super()._preprocess(X, **kwargs)
    
    if is_train:
        # Initialize and fit feature generator on training data
        self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
        self._feature_generator.fit(X=X)
    
    if self._feature_generator.features_in:
        # Convert categorical features to numeric via label encoding
        X = X.copy()
        X[self._feature_generator.features_in] = self._feature_generator.transform(X=X)
    
    # Handle missing values and convert to numpy
    return X.fillna(0).to_numpy(dtype=np.float32)
```

**Best Practice**: Import dependencies inside methods (not at module level) for modularity.

### Model Training

```python
def _fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
    # Import dependencies within method
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    
    # Select model based on problem type
    # Valid types: ['binary', 'multiclass', 'regression', 'quantile', 'softclass']
    if self.problem_type in ['regression', 'softclass']:
        model_cls = RandomForestRegressor
    else:
        model_cls = RandomForestClassifier
    
    # Preprocess training data
    X = self.preprocess(X, y=y, is_train=True)
    
    # Get hyperparameters and train
    params = self._get_model_params()
    self.model = model_cls(**params)
    self.model.fit(X, y)
```

### Default Hyperparameters

```python
def _set_default_params(self):
    default_params = {
        'n_estimators': 300,
        'n_jobs': -1,
        'random_state': 0,
    }
    for param, val in default_params.items():
        self._set_default_param_value(param, val)
```

### Supported Data Types

```python
def _get_default_auxiliary_params(self) -> dict:
    default_auxiliary_params = super()._get_default_auxiliary_params()
    extra_auxiliary_params = dict(
        # Raw dtypes: ['int', 'float', 'category', 'object', 'datetime']
        # 'object' includes text/images (specialized models only)
        # 'datetime' typically pre-converted to int
        valid_raw_types=['int', 'float', 'category'],
    )
    default_auxiliary_params.update(extra_auxiliary_params)
    return default_auxiliary_params
```

## Data Loading & Preparation

```python
from autogluon.tabular import TabularDataset

train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')
label = 'class'
train_data = train_data.sample(n=1000, random_state=0)
```

## Training Without TabularPredictor

### Label Cleaning

**Critical Step**: Convert string labels to numeric for binary classification.

```python
from autogluon.core.data import LabelCleaner
from autogluon.core.utils import infer_problem_type

# Separate features and labels
X = train_data.drop(columns=[label])
y = train_data[label]
X_test = test_data.drop(columns=[label])
y_test = test_data[label]

# Infer problem type and clean labels
problem_type = infer_problem_type(y=y)
label_cleaner = LabelCleaner.construct(problem_type=problem_type, y=y)
y_clean = label_cleaner.transform(y)
```

**Note**: `LabelCleaner` provides `inverse_transform()` to convert predictions back to original label format.

# Feature Cleaning & Model Training

## Feature Preprocessing

**Critical Step**: Convert object dtypes (strings) to categorical features using AutoGluon's feature generators.

```python
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from autogluon.common.utils.log_utils import set_logger_verbosity

set_logger_verbosity(2)
feature_generator = AutoMLPipelineFeatureGenerator()
X_clean = feature_generator.fit_transform(X)
```

**Important**: `AutoMLPipelineFeatureGenerator` does NOT:
- Fill missing values for numeric features
- Rescale numeric values
- One-hot encode categoricals

Add these operations in your `_preprocess` method if required.

## Model Training & Inference

### Fit Model

```python
custom_model = CustomRandomForestModel()
# Override defaults: CustomRandomForestModel(hyperparameters={'max_depth': 10})
custom_model.fit(X=X_clean, y=y_clean)

# Save/load model
# custom_model.save()
# custom_model = CustomRandomForestModel.load(path=load_path)
```

### Make Predictions

```python
# Prepare test data (apply same transformations)
X_test_clean = feature_generator.transform(X_test)
y_test_clean = label_cleaner.transform(y_test)

# Get predictions
y_pred = custom_model.predict(X_test_clean)

# Convert to original labels
y_pred_orig = label_cleaner.inverse_transform(y_pred)
```

### Evaluate Model

```python
score = custom_model.score(X_test_clean, y_test_clean)
print(f'Test score ({custom_model.eval_metric.name}) = {score}')
```

**Note**: Default eval_metric for binary classification is accuracy.

## Bagging Custom Models

**Quick Quality Improvement**: Bag models for better performance with minimal code.

```python
from autogluon.core.models import BaggedEnsembleModel

bagged_custom_model = BaggedEnsembleModel(CustomRandomForestModel())

# Required if model not in separate module (pickle serialization issue)
bagged_custom_model.params['fold_fitting_strategy'] = 'sequential_local'

bagged_custom_model.fit(X=X_clean, y=y_clean, k_fold=10)  # 10-fold bagging
bagged_score = bagged_custom_model.score(X_test_clean, y_test_clean)
```

**How it works**: Trains 10 models on different data splits; predictions are averaged.

## Training with TabularPredictor

### Basic Training

**Advantage**: Handles LabelCleaner, FeatureGenerator, and validation automatically.

```python
from autogluon.tabular import TabularPredictor

# Train 3 models with different hyperparameters
custom_hyperparameters = {
    CustomRandomForestModel: [
        {},  # defaults
        {'max_depth': 10},
        {'max_features': 0.9, 'max_depth': 20}
    ]
}

predictor = TabularPredictor(label=label).fit(
    train_data, 
    hyperparameters=custom_hyperparameters
)
```

### View Results & Predict

```python
# View model performance
predictor.leaderboard(test_data)

# Predict with best model (highest score_val)
y_pred = predictor.predict(test_data)

# Predict with specific model
# y_pred = predictor.predict(test_data, model='CustomRandomForestModel_3')
```

**Note**: `WeightedEnsemble` model automatically combines predictions for better scores.

## Hyperparameter Tuning

### Define Search Space

```python
from autogluon.common import space

custom_hyperparameters_hpo = {
    CustomRandomForestModel: {
        'max_depth': space.Int(lower=5, upper=30),
        'max_features': space.Real(lower=0.1, upper=1.0),
        'criterion': space.Categorical('gini', 'entropy'),
    }
}

# HPO for 20 seconds
predictor = TabularPredictor(label=label).fit(
    train_data,
    hyperparameters=custom_hyperparameters_hpo,
    hyperparameter_tune_kwargs='auto',  # enables HPO
    time_limit=20
)
```

### Analyze HPO Results

```python
leaderboard_hpo = predictor.leaderboard()

# Get best model hyperparameters
best_model_name = leaderboard_hpo[leaderboard_hpo['stack_level'] == 1]['model'].iloc[0]
predictor_info = predictor.info()
best_model_info = predictor_info['model_info'][best_model_name]

print(f'Best Model Hyperparameters ({best_model_name}):')
print(best_model_info['hyperparameters'])
```

**Identifier**: HPO trial models have suffix `/Tx` in their names.

## Training with Default Models

Combine custom model with AutoGluon's default models:

```python
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config

# Get default hyperparameters and add custom model
```

# Training Custom Models with Default Models

## Combining Custom and Default Models

**Key Technique**: Add tuned custom model to AutoGluon's default model set.

```python
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config

# Get default hyperparameters
custom_hyperparameters = get_hyperparameter_config('default')

# Add custom model with tuned hyperparameters
custom_hyperparameters[CustomRandomForestModel] = best_model_info['hyperparameters']

# Train default models + custom model
predictor = TabularPredictor(label=label).fit(
    train_data, 
    hyperparameters=custom_hyperparameters
)

# View results
predictor.leaderboard(test_data)
```

### Advanced: Multi-Layer Stack Ensemble

```python
# Use custom model in multi-layer stack ensemble
predictor = TabularPredictor(label=label).fit(
    train_data, 
    hyperparameters=custom_hyperparameters, 
    presets='best_quality'
)
```

## Summary

**Complete workflow for custom models in AutoGluon**:
1. Inherit from `AbstractModel`
2. Implement `_preprocess`, `_fit`, `_set_default_params`, `_get_default_auxiliary_params`
3. Test standalone or with bagging
4. Integrate with `TabularPredictor`
5. Perform hyperparameter tuning
6. Combine with default models

**Next Steps**:
- Submit custom models via [GitHub PR](https://github.com/autogluon/autogluon/pulls)
- See [Tabular Quick Start](../tabular-quick-start.ipynb) and [Tabular In-Depth](../tabular-indepth.ipynb)
- Advanced tutorial: [Custom Model Advanced](tabular-custom-model-advanced.ipynb)