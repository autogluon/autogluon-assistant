# Condensed: ```python

Summary: This tutorial covers AutoGluon's hyperparameter optimization (HPO) for tabular data, teaching how to define custom search spaces using `space.Real`, `space.Categorical`, and `space.Int` for neural networks and gradient boosting models. It demonstrates configuring `hyperparameter_tune_kwargs` with num_trials, scheduler, and searcher parameters, and passing model-specific hyperparameters (GBM, NN_TORCH) to `TabularPredictor.fit()`. Key techniques include validation data management with `tuning_data` and `holdout_frac`, time-limited training, and model evaluation using `predict_proba()` and `leaderboard()`. The tutorial emphasizes that HPO is generally unnecessary as AutoGluon's presets (like `presets="best_quality"`) typically outperform manual tuning, and warns about potential over-optimistic validation performance requiring separate test set evaluation.

*This is a condensed version that preserves essential implementation details and context.*

# AutoGluon Hyperparameter Tuning

## Installation & Setup

```python
!pip install autogluon.tabular[all]

from autogluon.tabular import TabularDataset, TabularPredictor

train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')
train_data = train_data.sample(n=1000, random_state=0)  # subsample for faster demo

label = 'occupation'
metric = 'accuracy'
```

## Hyperparameter Tuning

**⚠️ Warning:** Hyperparameter tuning is **NOT recommended** in most cases. AutoGluon performs best using presets like `presets="best_quality"` without tuning.

### Validation Data Considerations

- **Omit `tuning_data`** unless you have a strong reason (e.g., test data from different distribution)
- AutoGluon uses validation data for: HPO, early stopping, and ensemble construction
- Use `holdout_frac` to control validation split if needed
- **Caution:** Performance on validation data may be over-optimistic; always evaluate on separate test data

### Defining Hyperparameters

```python
from autogluon.common import space

nn_options = {
    'num_epochs': 10,  # training epochs (controls NN training time)
    'learning_rate': space.Real(1e-4, 1e-2, default=5e-4, log=True),  # log-scale search
    'activation': space.Categorical('relu', 'softrelu', 'tanh'),  # categorical HP
    'dropout_prob': space.Real(0.0, 0.5, default=0.1),  # real-valued HP
}

gbm_options = {
    'num_boost_round': 100,  # boosting rounds (controls GBM training time)
    'num_leaves': space.Int(lower=26, upper=66, default=36),  # integer HP
}

hyperparameters = {
    'GBM': gbm_options,
    'NN_TORCH': nn_options,  # Comment out if errors on Mac OSX
}
```

### Training with HPO

```python
time_limit = 2*60  # 2 minutes
num_trials = 5  # max HP configurations per model type

hyperparameter_tune_kwargs = {
    'num_trials': num_trials,
    'scheduler': 'local',
    'searcher': 'auto',  # random search with local scheduler
}

predictor = TabularPredictor(label=label, eval_metric=metric).fit(
    train_data,
    time_limit=time_limit,
    hyperparameters=hyperparameters,
    hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,  # HPO only if specified
)
```

### Evaluation

```python
# Predictions
predictor.predict_proba(test_data)

# Model performance comparison
predictor.leaderboard(test_data)
```

## Best Practices

- **Increase training resources** for better performance: larger `subsample_size`, higher `num_epochs`/`num_boost_round`, longer `time_limit`
- Use `verbosity=3` for detailed training output
- Unspecified hyperparameters use AutoGluon defaults (fixed or search spaces)
- See [Search Space documentation](../../../api/autogluon.common.space.rst) for more details