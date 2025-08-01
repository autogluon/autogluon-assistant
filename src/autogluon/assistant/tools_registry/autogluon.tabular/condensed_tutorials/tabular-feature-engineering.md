# Condensed: ```

Summary: This tutorial explains AutoGluon's automatic feature engineering capabilities for tabular data. It covers type-specific processing: numerical columns remain unchanged, categorical columns are integer-encoded, datetime columns are converted to numerical values with extracted temporal features (year, month, day, weekday), and text columns use either n-gram encoding or Transformer networks with MultiModal. The tutorial demonstrates how to implement custom feature generation pipelines using PipelineFeatureGenerator, CategoryFeatureGenerator, and IdentityFeatureGenerator classes, with examples of limiting categorical values and handling specific data types. It also addresses missing value handling and provides tips for customizing the feature engineering process.

*This is a condensed version that preserves essential implementation details and context.*

# Automatic Feature Engineering in AutoGluon

## Feature Processing by Data Type

### Numerical Columns
- No automated feature engineering for integer and floating point columns

### Categorical Columns
- Mapped to monotonically increasing integers for model compatibility

### Datetime Columns
- Converted to numerical Pandas datetime values
- Extracted features: `[year, month, day, dayofweek]` by default
- Missing/invalid values replaced with mean
- Note: Pandas datetime has min/max limits that may affect extreme dates

### Text Columns
- With MultiModal option: processed using Transformer neural networks
- Without MultiModal:
  - N-gram feature generation (creates "n-hot" encoded columns)
  - Special features (word counts, character counts, etc.)

## Additional Processing
- Single-value columns are dropped
- Duplicate columns are removed

## Implementation Example

```python
# Basic usage
predictor = TabularPredictor(label='class', problem_type='multiclass').fit(train_data)

# Custom feature generator
from autogluon.features.generators import PipelineFeatureGenerator, CategoryFeatureGenerator, IdentityFeatureGenerator
from autogluon.common.features.types import R_INT, R_FLOAT

mypipeline = PipelineFeatureGenerator(
    generators = [[        
        CategoryFeatureGenerator(maximum_num_cat=10),  # Limit categorical values
        IdentityFeatureGenerator(infer_features_in_args=dict(valid_raw_types=[R_INT, R_FLOAT])),
    ]]
)

# Use custom feature generator with predictor
predictor = TabularPredictor(label='label')
predictor.fit(df, hyperparameters={'GBM': {}}, feature_generator=mypipeline)
```

## Missing Value Handling
- Datetime columns: missing values replaced with mean
- Other columns: NaN values preserved

## Customization Tips
- Use `PipelineFeatureGenerator` for custom pipelines
- Configure individual generators with parameters like `maximum_num_cat`
- For categorical recognition, explicitly set column type: `df["col"] = df["col"].astype("category")`