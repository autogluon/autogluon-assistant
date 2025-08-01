Summary: This tutorial explains how to handle multiple label columns in AutoGluon MultiModal using two approaches: (1) converting mutually exclusive labels into a single combined column, or (2) training separate models for non-mutually exclusive labels. It provides code examples for both methods, emphasizing the need to properly manage label columns during training and prediction. Key implementation notes include time allocation across multiple models, excluding other label columns as features, and maintaining consistent preprocessing between training and inference. The tutorial helps with multi-label classification tasks in AutoGluon's MultiModal framework.

# Multiple Label Columns with AutoMM

AutoGluon MultiModal doesn't natively support multiple label columns. Here's how to handle this challenge in different scenarios.

## Scenario 1: Mutually Exclusive Labels

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
    test_data[f'{label}'] = (predictions == label).astype(int)
```

## Scenario 2: Non-Mutually Exclusive Labels

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

Note that you need to ensure other label columns are excluded from features, and adjust the time_limit parameter accordingly. If you have N label columns, consider allocating your total available time divided by N for each predictor

## Customization
To learn how to customize AutoMM, please refer to [Customize AutoMM](../advanced_topics/customization.ipynb).
