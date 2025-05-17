# Condensed: Adding a custom model to AutoGluon (Advanced)

Summary: This tutorial demonstrates advanced AutoGluon customization techniques, specifically focusing on implementing custom models and feature generators to maintain control over feature preprocessing. It shows how to prevent feature dropping through model-specific parameter overrides and custom feature generators, implement specialized feature handling using BulkFeatureGenerator, and configure feature metadata for custom preprocessing paths. The tutorial enables tasks like creating models that preserve unique-valued features and implementing custom preprocessing logic for specific features. Key functionalities include custom model class implementation, feature generator customization, metadata configuration, and integration with TabularPredictor, making it valuable for developers needing fine-grained control over AutoGluon's preprocessing pipeline.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details:

# Adding Custom Models to AutoGluon (Advanced)

## Key Concepts
- Demonstrates how to prevent feature dropping during preprocessing
- Shows implementation of custom feature generators
- Explains how to maintain specific features throughout the model pipeline

## Implementation Details

### 1. Force Features to Stay During Model-Specific Preprocessing

```python
class DummyModelKeepUnique(AbstractModel):
    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            drop_unique=False,  # Prevents dropping features with only 1 unique value
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params
```

### 2. Custom Feature Generator Implementation

```python
class CustomFeatureGeneratorWithUserOverride(BulkFeatureGenerator):
    def _get_default_generators(self, automl_generator_kwargs: dict = None):
        if automl_generator_kwargs is None:
            automl_generator_kwargs = dict()
            
        generators = [
            [
                # Normal features preprocessing
                AutoMLPipelineFeatureGenerator(
                    banned_feature_special_types=['user_override'], 
                    **automl_generator_kwargs
                ),
                # Special features preprocessing
                IdentityFeatureGenerator(
                    infer_features_in_args=dict(
                        required_special_types=['user_override']
                    )
                ),
            ],
        ]
        return generators
```

### 3. Feature Metadata Configuration

```python
feature_metadata = FeatureMetadata.from_df(train_data)
feature_metadata = feature_metadata.add_special_types({
    'feature1': ['user_override'],
    'feature2': ['user_override'],
})
```

### 4. Usage with TabularPredictor

```python
predictor = TabularPredictor(label=label)
predictor.fit(
    train_data=train_data,
    feature_metadata=feature_metadata,
    feature_generator=CustomFeatureGeneratorWithUserOverride(),
    hyperparameters={
        'GBM': {},
        DummyModelKeepUnique: {},
        # Alternative: DummyModel: {'ag_args_fit': {'drop_unique': False}}
    }
)
```

## Important Notes & Best Practices

1. Custom feature generator code must be in a separate Python file for serialization
2. Use `user_override` special type to mark features that need custom preprocessing
3. Can combine custom models with default AutoGluon models
4. Two ways to prevent feature dropping:
   - Model-specific: Override `_get_default_auxiliary_params`
   - Global: Create custom feature generator with separate preprocessing logic

## Warning
Custom model and feature generator classes must be defined in separate files from the main process for proper serialization.