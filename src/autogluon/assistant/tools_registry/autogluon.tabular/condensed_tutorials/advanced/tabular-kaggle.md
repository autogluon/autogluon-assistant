# Condensed: ```

Summary: This tutorial covers AutoGluon TabularPredictor implementation for Kaggle competitions, including merging multiple CSV files with pandas, configuring TabularPredictor with custom evaluation metrics (e.g., ROC-AUC), training models using `presets='best_quality'` for maximum accuracy, and generating predictions with `predict_proba()`. Key techniques include proper data joining strategies, class label verification (`positive_class`, `class_labels`), extracting single-class probabilities with `as_multiclass=False`, and creating Kaggle submission files. Helps with tasks like binary/multiclass classification workflows, handling multi-file datasets, model training with time limits, and competition submissions via Kaggle CLI. Covers critical parameters like `eval_metric`, `time_limit`, advanced options (`num_bag_folds`, `num_stack_levels`), and best practices for time-series data validation.

*This is a condensed version that preserves essential implementation details and context.*

# AutoGluon for Kaggle Competitions - Condensed Guide

## Data Loading and Preparation

```python
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor

directory = '~/IEEEfraud/'
label = 'isFraud'
eval_metric = 'roc_auc'  # Specify competition metric
save_path = directory + 'AutoGluonModels/'

train_identity = pd.read_csv(directory+'train_identity.csv')
train_transaction = pd.read_csv(directory+'train_transaction.csv')

# Join multiple CSV files into single table
train_data = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')
```

**⚠️ Important:** Use appropriate join strategy for your competition (left-join shown here).

## Model Training

```python
predictor = TabularPredictor(label=label, eval_metric=eval_metric, path=save_path, verbosity=3).fit(
    train_data, presets='best_quality', time_limit=3600
)

results = predictor.fit_summary()
```

**Key Parameter:** `presets='best_quality'` maximizes accuracy (increase `time_limit` for better results).

## Predictions and Submission

```python
# Load and join test data (same method as training)
test_identity = pd.read_csv(directory+'test_identity.csv')
test_transaction = pd.read_csv(directory+'test_transaction.csv')
test_data = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')

# Get predicted probabilities for positive class only
y_predproba = predictor.predict_proba(test_data, as_multiclass=False)

# Create submission file
submission = pd.read_csv(directory+'sample_submission.csv')
submission['isFraud'] = y_predproba
submission.to_csv(directory+'my_submission.csv', index=False)
```

### Class Label Verification

```python
# Binary classification
predictor.positive_class

# Multiclass classification
predictor.class_labels  # Corresponds to predict_proba() columns
```

**⚠️ Critical:** Verify predicted probabilities match Kaggle's expected class before submission.

### Submit to Kaggle
```bash
kaggle competitions submit -c ieee-fraud-detection -f sample_submission.csv -m "my first submission"
```

## Best Practices

1. **Evaluation Metric:** Always specify the competition's metric if provided
2. **Time-based Data:** Use recent training examples as validation set for time-series competitions
3. **Feature Engineering:** Focus efforts here rather than hyperparameter tuning
4. **Advanced Options:** Explore `num_bag_folds`, `num_stack_levels`, `num_bag_sets`, `hyperparameter_tune_kwargs` after mastering basics

## Troubleshooting

- Verify file permissions for Kaggle data access
- Check Kaggle account setup and [FAQ](https://www.kaggle.com/general/14438) for download/submission issues