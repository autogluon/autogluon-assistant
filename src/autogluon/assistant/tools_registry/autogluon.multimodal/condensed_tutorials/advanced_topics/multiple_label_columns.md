# Condensed: Multiple Label Columns in AutoMM 

Summary: This tutorial provides implementation techniques for handling multiple label columns in AutoGluon MultiModal, addressing a native limitation of the framework. It covers two approaches: (1) converting mutually exclusive label columns into a single combined label with preprocessing code for transformation and postprocessing for converting predictions back, and (2) training separate predictors for non-mutually exclusive labels where multiple categories can be true simultaneously. The tutorial includes complete Python code examples for both approaches, emphasizing the importance of feature selection and time management when training multiple predictors.

# Multiple Label Columns in AutoMM 

This tutorial explains how to handle multiple label columns with AutoGluon MultiModal.

## Problem Statement

AutoGluon MultiModal doesn't natively support multiple label columns. Here's how to handle this challenge in both frameworks.

## Option 1: Mutually Exclusive Labels

When your label columns are mutually exclusive (only one can be true at a time):

```python
# Preprocessing: Convert multiple columns to single label
def combine_labels(row, label_columns):
    for label in label_columns:
        if row[label] == 1:
            return label
    return 'none'

# Apply transformation
df['combined_label'] = df.apply(lambda row: combine_labels(row, label_columns), axis=1)

# For MultiModal
from autogluon.multimodal import MultiModalPredictor
predictor = MultiModalPredictor(label='combined_label').fit(df)

# Postprocessing (if needed): Convert predictions back to multiple columns
predictions = predictor.predict(test_data)
for label in label_columns:
    test_data[f'pred_{label}'] = (predictions == label).astype(int)
```

## Option 2: Non-Mutually Exclusive Labels

When your label columns are NOT mutually exclusive (multiple can be true simultaneously):

```python
# Define label columns
label_columns = ['label1', 'label2', 'label3']
predictors = {}

# For each label column
for label in label_columns:
    # Create copy without other label columns
    train_df = df.drop(columns=[l for l in label_columns if l != label])
    
    # For MultiModal
    from autogluon.multimodal import MultiModalPredictor
    predictors[label] = MultiModalPredictor(label=label).fit(train_df)

# Predict with each model
for label in label_columns:
    # Remove all label columns from test features
    test_features = test_data.drop(columns=label_columns)
    test_data[f'pred_{label}'] = predictors[label].predict(test_features)
```

## Important Notes
 
Ensure other label columns are excluded from features.

When training multiple predictors (Option 2), adjust the time_limit parameter accordingly. If you have N label columns, consider allocating your total available time divided by N for each predictor

This approach allows you to use both AutoGluon MultiModal with multiple label columns despite their native limitations.
