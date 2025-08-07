# Condensed: ```python

Summary: This tutorial demonstrates how to implement custom models in AutoGluon by extending the AbstractModel class, specifically creating a custom RandomForest implementation. It covers preprocessing with label encoding, dynamic model selection based on problem type, and proper integration with AutoGluon's ecosystem. Key functionalities include custom model training, feature preprocessing, model saving/loading, bagged ensembles for improved performance, and hyperparameter tuning using search spaces. The tutorial helps with tasks like implementing custom ML algorithms within AutoGluon, integrating models with TabularPredictor, and optimizing model performance through bagging and hyperparameter optimization.

*This is a condensed version that preserves essential implementation details and context.*

# Custom Model Implementation in AutoGluon

## Installation and Setup
```python
!pip install autogluon.tabular[all]
```

## Creating a Custom Random Forest Model

```python
import numpy as np
import pandas as pd
from autogluon.core.models import AbstractModel
from autogluon.features.generators import LabelEncoderFeatureGenerator

class CustomRandomForestModel(AbstractModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator = None

    def _preprocess(self, X: pd.DataFrame, is_train=False, **kwargs) -> np.ndarray:
        print(f'Entering the `_preprocess` method: {len(X)} rows of data (is_train={is_train})')
        X = super()._preprocess(X, **kwargs)

        if is_train:
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)
        if self._feature_generator.features_in:
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(X=X)
        return X.fillna(0).to_numpy(dtype=np.float32)

    def _fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        print('Entering the `_fit` method')
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        if self.problem_type in ['regression', 'softclass']:
            model_cls = RandomForestRegressor
        else:
            model_cls = RandomForestClassifier

        X = self.preprocess(X, is_train=True)
        params = self._get_model_params()
        print(f'Hyperparameters: {params}')
        self.model = model_cls(**params)
        self.model.fit(X, y)
        print('Exiting the `_fit` method')

    def _set_default_params(self):
        default_params = {
            'n_estimators': 300,
            'n_jobs': -1,
            'random_state': 0,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            valid_raw_types=['int', 'float', 'category'],
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params
```

## Loading the Data
```python
from autogluon.tabular import TabularDataset

train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')
label = 'class'
train_data = train_data.sample(n=1000, random_state=0)  # subsample for faster demo
```

## Training Without TabularPredictor

### Cleaning Labels
```python
# Separate features and labels
X = train_data.drop(columns=[label])
y = train_data[label]
X_test = test_data.drop(columns=[label])
y_test = test_data[label]

from autogluon.core.data import LabelCleaner
from autogluon.core.utils import infer_problem_type

# Convert labels to appropriate numeric format
problem_type = infer_problem_type(y=y)
label_cleaner = LabelCleaner.construct(problem_type=problem_type, y=y)
y_clean = label_cleaner.transform(y)
```

The implementation creates a custom Random Forest model by extending AutoGluon's AbstractModel class. Key components include:

1. **Preprocessing**: Handles categorical features via label encoding and missing values
2. **Model fitting**: Dynamically selects RandomForestClassifier or RandomForestRegressor based on problem type
3. **Default parameters**: Sets sensible defaults for the Random Forest algorithm
4. **Data type handling**: Specifies which data types the model can process

The example demonstrates how to use the custom model outside TabularPredictor for debugging purposes, including proper label cleaning for binary classification.

# Custom Model Integration in AutoGluon - Part 2

## Feature Cleaning

```python
from autogluon.common.utils.log_utils import set_logger_verbosity
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
set_logger_verbosity(2)

feature_generator = AutoMLPipelineFeatureGenerator()
X_clean = feature_generator.fit_transform(X)
```

The `AutoMLPipelineFeatureGenerator` converts string features to categorical types but doesn't:
- Fill missing values for numeric features
- Rescale numeric features
- One-hot encode categoricals

## Model Training and Prediction

```python
custom_model = CustomRandomForestModel()
# Optional: custom_model = CustomRandomForestModel(hyperparameters={'max_depth': 10})
custom_model.fit(X=X_clean, y=y_clean)

# Save/load functionality
# custom_model.save()
# custom_model = CustomRandomForestModel.load(path=load_path)
```

### Making Predictions

```python
# Prepare test data
X_test_clean = feature_generator.transform(X_test)
y_test_clean = label_cleaner.transform(y_test)

# Get predictions
y_pred = custom_model.predict(X_test_clean)

# Convert predictions back to original format
y_pred_orig = label_cleaner.inverse_transform(y_pred)
```

### Model Evaluation

```python
score = custom_model.score(X_test_clean, y_test_clean)
print(f'Test score ({custom_model.eval_metric.name}) = {score}')
```

## Bagged Ensemble Model

```python
from autogluon.core.models import BaggedEnsembleModel
bagged_custom_model = BaggedEnsembleModel(CustomRandomForestModel())

# Required if custom model isn't in a separate module
bagged_custom_model.params['fold_fitting_strategy'] = 'sequential_local'
bagged_custom_model.fit(X=X_clean, y=y_clean, k_fold=10)

bagged_score = bagged_custom_model.score(X_test_clean, y_test_clean)
print(f'Bagging increased model accuracy by {round(bagged_score - score, 4) * 100}%!')
```

## Integration with TabularPredictor

```python
from autogluon.tabular import TabularPredictor

# Train multiple versions with different hyperparameters
custom_hyperparameters = {
    CustomRandomForestModel: [
        {}, 
        {'max_depth': 10}, 
        {'max_features': 0.9, 'max_depth': 20}
    ]
}
predictor = TabularPredictor(label=label).fit(train_data, hyperparameters=custom_hyperparameters)

# View model performance
predictor.leaderboard(test_data)

# Make predictions
y_pred = predictor.predict(test_data)
# Optional: y_pred = predictor.predict(test_data, model='CustomRandomForestModel_3')
```

## Hyperparameter Tuning

```python
from autogluon.common import space
custom_hyperparameters_hpo = {
    CustomRandomForestModel: {
        'max_depth': space.Int(lower=5, upper=30),
        'max_features': space.Real(lower=0.1, upper=1.0),
        'criterion': space.Categorical('gini', 'entropy'),
    }
}

predictor = TabularPredictor(label=label).fit(
    train_data,
    hyperparameters=custom_hyperparameters_hpo,
    hyperparameter_tune_kwargs='auto',  # enables HPO
    time_limit=20
)

# View best hyperparameters
best_model_name = predictor.leaderboard()[predictor.leaderboard()['stack_level'] == 1]['model'].iloc[0]
best_model_info = predictor.info()['model_info'][best_model_name]
print(f'Best Model Hyperparameters: {best_model_info["hyperparameters"]}')
```

The tutorial demonstrates how to integrate custom models with AutoGluon's ecosystem, including feature preprocessing, model training, bagging for improved performance, and hyperparameter tuning.

# Custom Models in AutoGluon - Implementation Guide (Part 3/3)

## Adding Custom Models to AutoGluon

This section shows how to integrate a custom model with AutoGluon:

```python
# Add a custom RandomForest model with tuned hyperparameters
custom_hyperparameters = get_hyperparameter_config('default')
custom_hyperparameters[CustomRandomForestModel] = best_model_info['hyperparameters']

# Train with custom model alongside default models
predictor = TabularPredictor(label=label).fit(train_data, hyperparameters=custom_hyperparameters)
predictor.leaderboard(test_data)
```

## Wrapping Up

To add a custom model to AutoGluon:
1. Create a model class that inherits from AbstractModel
2. Implement required methods (fit, predict, etc.)
3. Add hyperparameter tuning capabilities
4. Integrate with AutoGluon's training pipeline

For advanced custom models, refer to the "Adding a custom model to AutoGluon (Advanced)" tutorial.