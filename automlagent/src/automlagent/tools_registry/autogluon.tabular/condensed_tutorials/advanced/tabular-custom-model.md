# Condensed: Adding a custom model to AutoGluon

Summary: This tutorial provides implementation guidance for adding custom models to AutoGluon, focusing on inheriting from the AbstractModel class and following AutoGluon's API patterns. It covers essential techniques for model preprocessing, fitting, and integration with TabularPredictor, including handling feature cleaning, model serialization, and hyperparameter tuning. Key functionalities include implementing custom RandomForest models, bagged ensembles, feature generation, and optimizing model performance through hyperparameter search spaces. The tutorial serves as a reference for tasks involving custom model integration, ensemble creation, and automated machine learning pipeline development within the AutoGluon framework, with specific examples of time limits, GPU support, and special data type handling.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed version focusing on the essential implementation details:

# Adding a Custom Model to AutoGluon

## Key Points
- Custom models must inherit from `AbstractModel` class
- Models need to follow AutoGluon's API to work with other models
- Implementation requires understanding of core functionality patterns

## Important Reference Models

| Feature | Reference Model |
|---------|----------------|
| Time limits & early stopping | `LGBModel`, `RFModel` |
| Memory usage limits | `LGBModel`, `RFModel` |
| Sample weights | `LGBModel` |
| GPU support | `LGBModel` |
| Non-serializable models | `NNFastAiTabularModel` |
| Special problem types | `RFModel` |
| Text features | `TextPredictorModel` |
| Image features | `ImagePredictorModel` |
| Custom HPO | `LGBModel` |

## Best Practices
1. Review base implementation in `AbstractModel` class
2. Study reference implementations for specific features
3. Ensure compatibility with AutoGluon's API
4. Follow proper inheritance patterns

## Implementation Prerequisites
- Understand AutoGluon basics from "Predicting Columns in a Table - Quick Start"
- Familiarity with model integration patterns
- Knowledge of desired model functionality requirements

```python
# Base import pattern
from autogluon.core.models.abstract.abstract_model import AbstractModel
```

## Reference Links
- AbstractModel source: `auto.gluon.ai/stable/_modules/autogluon/core/models/abstract/abstract_model.html`
- Default models documentation: `autogluon.tabular.models`

This condensed version maintains the critical implementation details while removing unnecessary explanatory text and focusing on the practical aspects of custom model implementation.

Here's the condensed tutorial content focusing on key implementation details:

# Custom Model Implementation in AutoGluon

## Key Implementation Details

### CustomRandomForestModel Class
```python
class CustomRandomForestModel(AbstractModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator = None
```

### Critical Methods

1. **_preprocess**
```python
def _preprocess(self, X: pd.DataFrame, is_train=False, **kwargs) -> np.ndarray:
    X = super()._preprocess(X, **kwargs)
    if is_train:
        self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
        self._feature_generator.fit(X=X)
    if self._feature_generator.features_in:
        X = X.copy()
        X[self._feature_generator.features_in] = self._feature_generator.transform(X=X)
    return X.fillna(0).to_numpy(dtype=np.float32)
```

2. **_fit**
```python
def _fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    
    model_cls = RandomForestRegressor if self.problem_type in ['regression', 'softclass'] else RandomForestClassifier
    X = self.preprocess(X, is_train=True)
    params = self._get_model_params()
    self.model = model_cls(**params)
    self.model.fit(X, y)
```

### Important Configurations

1. **Default Parameters**
```python
def _set_default_params(self):
    default_params = {
        'n_estimators': 300,
        'n_jobs': -1,
        'random_state': 0,
    }
```

2. **Auxiliary Parameters**
```python
def _get_default_auxiliary_params(self):
    return {
        'valid_raw_types': ['int', 'float', 'category']
    }
```

## Data Preparation

```python
# Load and clean data
train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
X = train_data.drop(columns=[label])
y = train_data[label]

# Clean labels
problem_type = infer_problem_type(y=y)
label_cleaner = LabelCleaner.construct(problem_type=problem_type, y=y)
y_clean = label_cleaner.transform(y)
```

## Best Practices
1. Import dependencies inside methods for modularity
2. Always call preprocess on data during fit and predict
3. Handle missing values appropriately (using fillna or model-specific handling)
4. Implement proper label cleaning for classification tasks
5. Specify valid data types in auxiliary parameters

Here's the condensed tutorial focusing on key implementation details and practices:

# Custom Model Implementation in AutoGluon - Part 3

## Feature Cleaning
```python
from autogluon.features.generators import AutoMLPipelineFeatureGenerator

# Clean features using AutoGluon's feature generator
feature_generator = AutoMLPipelineFeatureGenerator()
X_clean = feature_generator.fit_transform(X)
```

**Note**: AutoMLPipelineFeatureGenerator doesn't handle:
- Missing value imputation for numeric features
- Feature scaling
- One-hot encoding for categoricals

## Model Training and Prediction

```python
# Train model
custom_model = CustomRandomForestModel()
custom_model.fit(X=X_clean, y=y_clean)

# Save/Load model
# custom_model.save()
# custom_model = CustomRandomForestModel.load(path=load_path)

# Prepare and predict test data
X_test_clean = feature_generator.transform(X_test)
y_test_clean = label_cleaner.transform(y_test)
y_pred = custom_model.predict(X_test_clean)

# Get interpretable predictions
y_pred_orig = label_cleaner.inverse_transform(y_pred)
```

## Bagged Ensemble Implementation

```python
from autogluon.core.models import BaggedEnsembleModel

bagged_custom_model = BaggedEnsembleModel(CustomRandomForestModel())
bagged_custom_model.params['fold_fitting_strategy'] = 'sequential_local'  # Required for class not in separate module
bagged_custom_model.fit(X=X_clean, y=y_clean, k_fold=10)
```

## Integration with TabularPredictor

```python
from autogluon.tabular import TabularPredictor

# Train multiple models with different hyperparameters
custom_hyperparameters = {
    CustomRandomForestModel: [
        {}, 
        {'max_depth': 10}, 
        {'max_features': 0.9, 'max_depth': 20}
    ]
}
predictor = TabularPredictor(label=label).fit(train_data, hyperparameters=custom_hyperparameters)
```

## Hyperparameter Tuning

```python
from autogluon.common import space

# Define hyperparameter search space
custom_hyperparameters_hpo = {
    CustomRandomForestModel: {
        'max_depth': space.Int(lower=5, upper=30),
        'max_features': space.Real(lower=0.1, upper=1.0),
        'criterion': space.Categorical('gini', 'entropy'),
    }
}

# Train with HPO
predictor = TabularPredictor(label=label).fit(
    train_data,
    hyperparameters=custom_hyperparameters_hpo,
    hyperparameter_tune_kwargs='auto',
    time_limit=20
)
```

### Key Best Practices:
1. Use feature generators for consistent data preprocessing
2. Implement model saving/loading for production use
3. Consider using bagged ensembles for improved performance
4. Leverage TabularPredictor for advanced functionality
5. Use hyperparameter tuning for optimizing model performance

Here's the condensed version of the final chunk focusing on key implementation details:

# Adding Custom Model with Tuned Hyperparameters

## Key Implementation

1. Add tuned custom model to default models:
```python
# Add custom model with optimized hyperparameters
custom_hyperparameters = get_hyperparameter_config('default')
custom_hyperparameters[CustomRandomForestModel] = best_model_info['hyperparameters']
```

2. Train predictor with custom configuration:
```python
predictor = TabularPredictor(label=label).fit(
    train_data, 
    hyperparameters=custom_hyperparameters
)
```

## Important Variations

- For enhanced performance, use with stack ensembles:
```python
predictor = TabularPredictor(label=label).fit(
    train_data, 
    hyperparameters=custom_hyperparameters,
    presets='best_quality'  # Enables multi-layer stack ensemble
)
```

## Best Practices

1. Evaluate model performance:
```python
predictor.leaderboard(test_data)
```

2. Consider contributing custom models via PR to AutoGluon's repository

## Additional Resources
- Basic tutorials: 
  - "Predicting Columns in a Table - Quick Start"
  - "Predicting Columns in a Table - In Depth"
- Advanced custom models: "Adding a custom model to AutoGluon (Advanced)"

This concludes the implementation of custom models in AutoGluon, showing how to integrate optimized custom models with AutoGluon's existing framework.