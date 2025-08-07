# Condensed: ```

Summary: This tutorial demonstrates how to use AutoGluon for fraud detection in the IEEE Fraud detection competition. AutoGluon's TabularPredictor can be used to train a model for binary classification with minimal code. It shows how to use TabularPredictor for binary classification with TabularPredictor for binary classification with TabularPredictor for binary classification.

*This is a condensed version that preserves essential implementation details and context.*

# AutoGluon for Kaggle Competitions: IEEE Fraud Detection

## Data Setup and Preparation

```python
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor

directory = '~/IEEEfraud/'
label = 'isFraud'
eval_metric = 'roc_auc'
save_path = directory + 'AutoGluonModels/'

# Load and merge data
train_identity = pd.read_csv(directory+'train_identity.csv')
train_transaction = pd.read_csv(directory+'train_transaction.csv')
train_data = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')
```

## Training the Model

```python
# Train with best quality preset
predictor = TabularPredictor(label=label, eval_metric=eval_metric, path=save_path, verbosity=3).fit(
    train_data, presets='best_quality', time_limit=3600
)

results = predictor.fit_summary()
```

## Making Predictions

```python
# Load and merge test data
test_identity = pd.read_csv(directory+'test_identity.csv')
test_transaction = pd.read_csv(directory+'test_transaction.csv')
test_data = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')

# Get prediction probabilities for positive class
y_predproba = predictor.predict_proba(test_data, as_multiclass=False)
```

## Important: Verify Prediction Class Labels

```python
# For binary classification
predictor.positive_class

# For multiclass classification
predictor.class_labels  # classes correspond to columns of predict_proba() output
```

## Creating Submission File

```python
submission = pd.read_csv(directory+'sample_submission.csv')
submission['isFraud'] = y_predproba
submission.to_csv(directory+'my_submission.csv', index=False)
```

## Performance Optimization Tips

- Specify the appropriate evaluation metric from the competition
- For time-based data, reserve recent examples as validation data
- Use `presets='best_quality'` and focus on feature engineering
- Advanced options: `num_bag_folds`, `num_stack_levels`, `num_bag_sets`, `hyperparameter_tune_kwargs`, `hyperparameters`, `refit_full`

## Submission Command

```
kaggle competitions submit -c ieee-fraud-detection -f sample_submission.csv -m "my first submission"
```