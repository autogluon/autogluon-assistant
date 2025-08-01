# Condensed: ```python

Summary: This tutorial demonstrates how to prevent AutoGluon from dropping specific features during preprocessing. It covers three key techniques: (1) creating custom models with the `drop_unique=False` parameter to preserve single-value features, (2) implementing a `CustomFeatureGeneratorWithUserOverride` that uses `IdentityFeatureGenerator` to bypass preprocessing for tagged features, and (3) tagging features with special types like 'user_override' to control their processing. These methods are valuable for maintaining important features in machine learning pipelines, ensuring model interpretability, and handling domain-specific features that shouldn't undergo standard transformations.

*This is a condensed version that preserves essential implementation details and context.*

# Forcing Features to be Passed to Models Without Preprocessing/Dropping

## Setup and Data Loading

```python
!pip install autogluon.tabular[all]

from autogluon.tabular import TabularDataset

train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')
label = 'class'
train_data = train_data.sample(n=1000, random_state=0)  # subsample for faster demo
```

## Preventing Feature Dropping

### 1. Model-Specific Preprocessing

To prevent models from dropping features with only one unique value:

```python
from autogluon.core.models import AbstractModel

class DummyModel(AbstractModel):
    def _fit(self, X, **kwargs):
        print(f'Before {self.__class__.__name__} Preprocessing ({len(X.columns)} features):\n\t{list(X.columns)}')
        X = self.preprocess(X)
        print(f'After {self.__class__.__name__} Preprocessing ({len(X.columns)} features):\n\t{list(X.columns)}')
        print(X.head(5))

class DummyModelKeepUnique(DummyModel):
    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            drop_unique=False,  # Whether to drop features that have only 1 unique value
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params
```

### 2. Global Preprocessing Control

Create a custom feature generator to handle special features:

```python
# WARNING: Put this code in a separate python file for serialization
from autogluon.features import BulkFeatureGenerator, AutoMLPipelineFeatureGenerator, IdentityFeatureGenerator

class CustomFeatureGeneratorWithUserOverride(BulkFeatureGenerator):
    def __init__(self, automl_generator_kwargs: dict = None, **kwargs):
        generators = self._get_default_generators(automl_generator_kwargs=automl_generator_kwargs)
        super().__init__(generators=generators, **kwargs)

    def _get_default_generators(self, automl_generator_kwargs: dict = None):
        if automl_generator_kwargs is None:
            automl_generator_kwargs = dict()

        generators = [
            [
                # Normal features preprocessing
                AutoMLPipelineFeatureGenerator(banned_feature_special_types=['user_override'], **automl_generator_kwargs),

                # Skip preprocessing for special features
                IdentityFeatureGenerator(infer_features_in_args=dict(required_special_types=['user_override'])),
            ],
        ]
        return generators
```

### 3. Tagging Features for Special Handling

```python
# Add dummy feature to demonstrate preservation
train_data['dummy_feature'] = 'dummy value'
test_data['dummy_feature'] = 'dummy value'

from autogluon.tabular import FeatureMetadata
feature_metadata = FeatureMetadata.from_df(train_data)

# Tag features to preserve
feature_metadata = feature_metadata.add_special_types({
    'age': ['user_override'],
    'native-country': ['user_override'],
    'dummy_feature': ['user_override'],
})
```

## Implementation Example

```python
# Prepare data
X = train_data.drop(columns=[label])
y = train_data[label]
X_test = test_data.drop(columns=[label])
y_test = test_data[label]

# Preprocess labels
from autogluon.core.data import LabelCleaner
from autogluon.core.utils import infer_problem_type
problem_type = infer_problem_type(y=y)
label_cleaner = LabelCleaner.construct(problem_type=problem_type, y=y)
y_preprocessed = label_cleaner.transform(y)
y_test_preprocessed = label_cleaner.transform(y_test)

# Apply custom feature generator
my_custom_feature_generator = CustomFeatureGeneratorWithUserOverride(feature_metadata_in=feature_metadata)
X_preprocessed = my_custom_feature_generator.fit_transform(X)
X_test_preprocessed = my_custom_feature_generator.transform(X_test)
```

## Using with TabularPredictor

```python
# NOTE: This code must be in a separate file for serialization
from autogluon.tabular import TabularPredictor

feature_generator = CustomFeatureGeneratorWithUserOverride()
predictor = TabularPredictor(label=label)
predictor.fit(
    train_data=train_data,
    feature_metadata=feature_metadata,
    feature_generator=feature_generator,
    hyperparameters={
        'GBM': {},
        DummyModelKeepUnique: {},
        # Alternative: DummyModel: {'ag_args_fit': {'drop_unique': False}}
    }
)
```