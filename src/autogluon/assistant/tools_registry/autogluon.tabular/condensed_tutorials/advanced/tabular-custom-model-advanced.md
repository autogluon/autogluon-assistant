# Condensed: ```python

Summary: This tutorial teaches how to prevent AutoGluon from automatically dropping features during preprocessing, particularly those with single unique values. It covers two approaches: (1) overriding `_get_default_auxiliary_params()` in custom models to set `drop_unique=False` for model-specific preservation, and (2) creating custom feature generators using `BulkFeatureGenerator` with bifurcated preprocessing logic that applies different handling to features tagged with special types via `FeatureMetadata.add_special_types()`. Key implementations include using `IdentityFeatureGenerator` for no-op preprocessing, `AutoMLPipelineFeatureGenerator` with `banned_feature_special_types`, and integrating custom feature generators with `TabularPredictor`. This helps with tasks requiring mandatory columns for custom model logic, fine-tuning pre-trained models with specific feature requirements, and controlling feature engineering pipelines in AutoGluon workflows.

*This is a condensed version that preserves essential implementation details and context.*

# Force Features Through Preprocessing Without Dropping

## Overview
Preserve specific features in AutoGluon models even when they would normally be dropped (e.g., single unique value). Useful for models requiring specific columns for custom logic.

## Setup
```python
!pip install autogluon.tabular[all]

from autogluon.tabular import TabularDataset
train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')
label = 'class'
train_data = train_data.sample(n=1000, random_state=0)
```

## Method 1: Prevent Model-Specific Dropping

Override `_get_default_auxiliary_params()` in custom models:

```python
from autogluon.core.models import AbstractModel

class DummyModelKeepUnique(AbstractModel):
    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            drop_unique=False,  # Prevent dropping features with 1 unique value (default: True)
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params
```

## Method 2: Prevent Global Preprocessing Dropping

**⚠️ WARNING:** Must be in separate Python file for serialization.

Create custom feature generator with bifurcated preprocessing:

```python
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
                # Standard preprocessing for normal features
                AutoMLPipelineFeatureGenerator(banned_feature_special_types=['user_override'], **automl_generator_kwargs),
                
                # No-op preprocessing for user_override features
                IdentityFeatureGenerator(infer_features_in_args=dict(required_special_types=['user_override'])),
            ],
        ]
        return generators
```

### Tag Features with Special Type

```python
train_data['dummy_feature'] = 'dummy value'
test_data['dummy_feature'] = 'dummy value'

from autogluon.tabular import FeatureMetadata
feature_metadata = FeatureMetadata.from_df(train_data)

feature_metadata = feature_metadata.add_special_types(
    {
        'age': ['user_override'],
        'native-country': ['user_override'],
        'dummy_feature': ['user_override'],
    }
)
```

### Apply Custom Preprocessing

```python
from autogluon.core.data import LabelCleaner
from autogluon.core.utils import infer_problem_type

X = train_data.drop(columns=[label])
y = train_data[label]

# Preprocess labels
problem_type = infer_problem_type(y=y)
label_cleaner = LabelCleaner.construct(problem_type=problem_type, y=y)
y_preprocessed = label_cleaner.transform(y)

# Apply custom feature generator with metadata
my_custom_feature_generator = CustomFeatureGeneratorWithUserOverride(feature_metadata_in=feature_metadata)
X_preprocessed = my_custom_feature_generator.fit_transform(X)
```

## Using with TabularPredictor

```python
from autogluon.tabular import TabularPredictor

feature_generator = CustomFeatureGeneratorWithUserOverride()
predictor = TabularPredictor(label=label)
predictor.fit(
    train_data=train_data,
    feature_metadata=feature_metadata,  # With user_override tags
    feature_generator=feature_generator,  # Custom generator
    hyperparameters={
        'GBM': {},
        DummyModelKeepUnique: {},  # Uses _get_default_auxiliary_params
        # Alternative: DummyModel: {'ag_args_fit': {'drop_unique': False}},
    }
)
```

## Key Parameters
- **`drop_unique`**: Controls dropping of single-value features (default: `True`)
- **`banned_feature_special_types`**: Excludes features with specific tags from preprocessing
- **`required_special_types`**: Targets only features with specific tags