# Condensed: Multiple Label Columns with AutoMM

Summary: This tutorial provides workarounds for handling multiple label columns in AutoGluon MultiModalPredictor, which doesn't natively support multi-label scenarios. It covers two techniques: (1) combining mutually exclusive labels into a single column using a helper function, then converting predictions back to multi-column format, and (2) training separate MultiModalPredictor models for non-mutually exclusive labels (multi-label classification) by iteratively dropping other label columns from training data. Key implementation details include proper feature preprocessing to exclude label columns during prediction, time budget allocation across multiple models, and bidirectional conversion between single-label and multi-label formats. Useful for multi-class and multi-label classification tasks with AutoGluon.

*This is a condensed version that preserves essential implementation details and context.*

# Multiple Label Columns with AutoMM

AutoGluon MultiModal doesn't natively support multiple label columns. Handle this using two approaches:

## Mutually Exclusive Labels
When only one label can be true at a time, combine into single column:

```python
# Convert multiple columns to single label
def combine_labels(row, label_columns):
    for label in label_columns:
        if row[label] == 1:
            return label
    return 'none'

df['combined_label'] = df.apply(lambda row: combine_labels(row, label_columns), axis=1)

# Train
from autogluon.multimodal import MultiModalPredictor
predictor = MultiModalPredictor(label='combined_label').fit(df)

# Convert predictions back to multiple columns
predictions = predictor.predict(test_data)
for label in label_columns:
    test_data[f'{label}'] = (predictions == label).astype(int)
```

## Non-Mutually Exclusive Labels
When multiple labels can be true simultaneously, train separate models:

```python
label_columns = ['label1', 'label2', 'label3']
predictors = {}

# Train one model per label
for label in label_columns:
    train_df = df.drop(columns=[l for l in label_columns if l != label])
    predictors[label] = MultiModalPredictor(label=label).fit(train_df)

# Predict with each model
for label in label_columns:
    test_features = test_data.drop(columns=label_columns)
    test_data[f'pred_{label}'] = predictors[label].predict(test_features)
```

**Important:** Exclude other label columns from features. With N labels, divide total `time_limit` by N for each predictor.

See [Customize AutoMM](../advanced_topics/customization.ipynb) for additional customization options.