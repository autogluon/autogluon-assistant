# Condensed: ```

Summary: This tutorial covers AutoGluon's automatic feature engineering pipeline for tabular data preprocessing. It demonstrates how to use `AutoMLPipelineFeatureGenerator` to transform different data types: categorical columns to integers, datetime columns to numerical values plus extracted features (year/month/day/dayofweek), and text columns to n-gram and statistical features. Key implementations include explicit categorical marking with `.astype("category")`, handling missing values (NaNs retained for most types, mean-imputed for datetime), and custom pipeline creation using `PipelineFeatureGenerator` with configurable generators like `CategoryFeatureGenerator(maximum_num_cat=10)` and `TextNgramFeatureGenerator`. Useful for building ML preprocessing pipelines, handling mixed data types, and customizing feature transformation strategies in AutoGluon tabular workflows.

*This is a condensed version that preserves essential implementation details and context.*

# Automatic Feature Engineering

## Feature Type Processing

### Numerical Columns
No automated feature engineering applied to integer or floating point columns.

### Categorical Columns
Mapped to monotonically increasing integers for model compatibility.

### Datetime Columns
Converted into:
- Numerical Pandas datetime (constrained by `pandas.Timestamp.min/max`)
- Extracted columns: `[year, month, day, dayofweek]` (default, configurable via `DatetimeFeatureGenerator`)

**Note:** Missing, invalid, or out-of-range values converted to mean of valid rows.

### Text Columns
Two processing modes:
1. **MultiModal enabled:** Full Transformer neural network with pretrained NLP models
2. **Default processing:**
   - **N-gram features:** Extracts n-grams (default: individual words), creates n-hot encoded columns (configurable via `TextNgramFeatureGenerator`)
   - **Numerical features:** Word counts, character counts, uppercase proportion, etc. (configurable via `TextSpecialFeatureGenerator`)

### Additional Processing
- Single-value columns dropped
- Duplicate columns removed

## Implementation Example

```python
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
import pandas as pd

# Create sample data with multiple types
dfx['B'] = dfx['B'].astype(int)  # Integer
dfx['C'] = datetime(2000,1,1) + pd.to_timedelta(dfx['C'].astype(int), unit='D')  # Datetime
dfx['D'] = pd.cut(dfx['D'] * 10, [-np.inf,-5,0,5,np.inf], labels=['v','w','x','y'])  # Categorical
dfx['E'] = pd.Series([...])  # Text

# Apply default feature generation
auto_ml_pipeline_feature_generator = AutoMLPipelineFeatureGenerator()
auto_ml_pipeline_feature_generator.fit_transform(X=dfx)
```

**Results:**
- Float/int columns unchanged
- Datetime → raw value (nanoseconds) + year, month, day, dayofweek
- Categorical strings → integers
- Text → summary features (char_count, etc.) + n-hot word matrix

### Explicit Categorical Marking

```python
# Mark integer column as categorical
dfx["B"] = dfx["B"].astype("category")
```

## Missing Value Handling

```python
dfx.iloc[0] = np.nan  # Set first row to NaN
auto_ml_pipeline_feature_generator.fit_transform(X=dfx)
```

**Behavior:**
- Float, int, categorical, text: NaNs retained
- Datetime: NaNs replaced with mean of non-NaN values

## Custom Feature Engineering

Use `PipelineFeatureGenerator` with custom parameters:

```python
from autogluon.features.generators import PipelineFeatureGenerator, CategoryFeatureGenerator, IdentityFeatureGenerator
from autogluon.common.features.types import R_INT, R_FLOAT

mypipeline = PipelineFeatureGenerator(
    generators = [[        
        CategoryFeatureGenerator(maximum_num_cat=10),  # Limit categories, rare → NaN
        IdentityFeatureGenerator(infer_features_in_args=dict(valid_raw_types=[R_INT, R_FLOAT])),
    ]]
)
mypipeline.fit_transform(X=dfx)
```

**See:** `examples/tabular/example_custom_feature_generator.py` for advanced customization.