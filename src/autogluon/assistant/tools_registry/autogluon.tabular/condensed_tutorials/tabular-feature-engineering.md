# Condensed: AutoGluon Tabular - Feature Engineering

Summary: This tutorial covers AutoGluon's tabular feature engineering implementation, focusing on automatic data type detection and processing for boolean, categorical, numerical, datetime, and text columns. It demonstrates how to implement custom feature processing pipelines, configure data type overrides, and handle automated feature engineering for datetime (extracting year, month, day components) and text data (using either Transformer networks or n-gram generation). Key functionalities include automatic column type detection rules, missing value handling, and categorical encoding. The tutorial helps with tasks like building custom feature pipelines, optimizing datetime processing, and implementing text feature generation, while highlighting best practices for data type management and column preprocessing.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed version focusing on key implementation details and practices:

# AutoGluon Tabular - Feature Engineering

## Core Column Types
```python
- boolean: [A, B]
- numerical: [1.3, 2.0, -1.6]
- categorical: [Red, Blue, Yellow]
- datetime: [1/31/2021, Mar-31]
- text: [longer text content]
- image: [path/image123.png] (with MultiModal option)
```

## Key Detection Rules
- **Boolean**: Columns with exactly 2 unique values
- **Categorical**: String columns not classified as text
- **Numerical**: Passed through unchanged (float/int)
- **Text**: Most rows unique + multiple words per row
- **Datetime**: Auto-detected via Pandas datetime conversion

## Important Configurations

```python
# Override problem type detection
predictor = TabularPredictor(
    label='class', 
    problem_type='multiclass'
).fit(train_data)
```

## Automatic Feature Engineering

### Datetime Processing
- Converts to numerical timestamp
- Extracts: `[year, month, day, dayofweek]`
- Missing values → column mean

### Text Processing
Two approaches:
1. With MultiModal: Uses Transformer neural network
2. Standard:
   - N-gram feature generation
   - Special features (word counts, character counts, etc.)

## Implementation Example

```python
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from autogluon.tabular import TabularDataset, TabularPredictor

# Basic usage
auto_ml_pipeline = AutoMLPipelineFeatureGenerator()
transformed_data = auto_ml_pipeline.fit_transform(X=data)

# Custom pipeline
from autogluon.features.generators import PipelineFeatureGenerator, CategoryFeatureGenerator
custom_pipeline = PipelineFeatureGenerator(
    generators = [[        
        CategoryFeatureGenerator(maximum_num_cat=10),
        IdentityFeatureGenerator(
            infer_features_in_args=dict(
                valid_raw_types=[R_INT, R_FLOAT]
            )
        ),
    ]]
)
```

## Best Practices
1. Explicitly mark categorical columns: `df['col'].astype('category')`
2. Configure datetime feature extraction as needed
3. Use MultiModal option for complex text processing
4. Consider custom pipelines for specific requirements
5. Handle missing values appropriately per column type

## Critical Notes
- Duplicate columns are automatically removed
- Single-value columns are dropped
- Datetime extremes are bounded by Pandas Timestamp limits
- Categorical features are encoded as integers for model compatibility