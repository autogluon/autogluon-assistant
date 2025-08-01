# Condensed: Multiple Label Columns with AutoMM

Summary: This tutorial explains how to handle multiple label columns in AutoGluon MultiModal using two approaches: (1) converting mutually exclusive labels into a single combined column, or (2) training separate models for non-mutually exclusive labels. It provides code examples for both methods, emphasizing the need to properly manage label columns during training and prediction. Key implementation notes include time allocation across multiple models, excluding other label columns as features, and maintaining consistent preprocessing between training and inference. The tutorial helps with multi-label classification tasks in AutoGluon's MultiModal framework.

*This is a condensed version that preserves essential implementation details and context.*

# Multiple Label Columns with AutoMM

## Multiple Label Columns with AutoMM

### Implementation Details

When working with multiple label columns in AutoGluon MultiModal, you have two main approaches:

1. **Mutually Exclusive Labels**: Convert multiple columns to a single label
   ```python
   df['combined_label'] = df.apply(lambda row: combine_labels(row, label_columns), axis=1)
   predictor = MultiModalPredictor(label='combined_label').fit(df)
   ```

2. **Non-Mutually Exclusive Labels**: Train separate models
   ```python
   predictors = {}
   for label in label_columns:
       train_df = df.drop(columns=[l for l in label_columns if l != label])
       predictors[label] = MultiModalPredictor(label=label).fit(train_df)
   ```

### Key Implementation Notes

- For mutually exclusive labels, convert to a single column before training
- For non-mutually exclusive labels, train separate models for each label
- When using multiple predictors, allocate time_limit/N for each of N label columns
- Always exclude other label columns from features when training individual models
- For prediction, apply the same preprocessing/postprocessing steps as during training

### Best Practices

- Ensure consistent preprocessing between training and inference
- When using the multiple-model approach, be aware of increased training time and resource usage
- Consider model ensembling if appropriate for your use case
- For customization options, refer to the AutoMM customization documentation