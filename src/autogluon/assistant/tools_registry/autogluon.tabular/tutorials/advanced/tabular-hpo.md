Summary: This tutorial covers AutoGluon's hyperparameter optimization (HPO) for tabular data, teaching how to define custom search spaces using `space.Real`, `space.Categorical`, and `space.Int` for neural networks and gradient boosting models. It demonstrates configuring `hyperparameter_tune_kwargs` with num_trials, scheduler, and searcher parameters, and passing model-specific hyperparameters (GBM, NN_TORCH) to `TabularPredictor.fit()`. Key techniques include validation data management with `tuning_data` and `holdout_frac`, time-limited training, and model evaluation using `predict_proba()` and `leaderboard()`. The tutorial emphasizes that HPO is generally unnecessary as AutoGluon's presets (like `presets="best_quality"`) typically outperform manual tuning, and warns about potential over-optimistic validation performance requiring separate test set evaluation.

```python
!pip install autogluon.tabular[all]

```


```python
from autogluon.tabular import TabularDataset, TabularPredictor

train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')
subsample_size = 1000  # subsample data for a faster demo
train_data = train_data.sample(n=subsample_size, random_state=0)

label = 'occupation'
metric = 'accuracy'
```

## Specifying hyperparameters and tuning them

**Note: We don't recommend doing hyperparameter-tuning with AutoGluon in most cases**. AutoGluon achieves its best performance without hyperparameter tuning and simply specifying one of the available presets, such as `presets="best_quality"`.

We demonstrate hyperparameter-tuning and how you can provide your own validation dataset that AutoGluon internally relies on to: tune hyperparameters, early-stop iterative training, and construct model ensembles. One reason you may specify validation data is when future test data will stem from a different distribution than training data (and your specified validation data is more representative of the future data that will likely be encountered).

 If you don't have a strong reason to provide your own validation dataset, we recommend you omit the `tuning_data` argument. This lets AutoGluon automatically select validation data from your provided training set (it uses smart strategies such as stratified sampling).  For greater control, you can specify the `holdout_frac` argument to tell AutoGluon what fraction of the provided training data to hold out for validation.

**Caution:** Since AutoGluon tunes internal knobs based on this validation data, performance estimates reported on this data may be over-optimistic. For unbiased performance estimates, you should always call `predict()` on a separate dataset (that was never passed to `fit()`), as we did in the previous **Quick-Start** tutorial. We also emphasize that most options specified in this tutorial are chosen to minimize runtime for the purposes of demonstration and you should select more reasonable values in order to obtain high-quality models.

`fit()` trains neural networks and various types of tree ensembles by default. You can specify various hyperparameter values for each type of model. For each hyperparameter, you can either specify a single fixed value, or a search space of values to consider during hyperparameter optimization. Hyperparameters which you do not specify are left at default settings chosen automatically by AutoGluon, which may be fixed values or search spaces.

Refer to the [Search Space documentation](../../../api/autogluon.common.space.rst) to learn more about AutoGluon search space.


```python
from autogluon.common import space

nn_options = {  # specifies non-default hyperparameter values for neural network models
    'num_epochs': 10,  # number of training epochs (controls training time of NN models)
    'learning_rate': space.Real(1e-4, 1e-2, default=5e-4, log=True),  # learning rate used in training (real-valued hyperparameter searched on log-scale)
    'activation': space.Categorical('relu', 'softrelu', 'tanh'),  # activation function used in NN (categorical hyperparameter, default = first entry)
    'dropout_prob': space.Real(0.0, 0.5, default=0.1),  # dropout probability (real-valued hyperparameter)
}

gbm_options = {  # specifies non-default hyperparameter values for lightGBM gradient boosted trees
    'num_boost_round': 100,  # number of boosting rounds (controls training time of GBM models)
    'num_leaves': space.Int(lower=26, upper=66, default=36),  # number of leaves in trees (integer hyperparameter)
}

hyperparameters = {  # hyperparameters of each model type
    'GBM': gbm_options,
    'NN_TORCH': nn_options,  # NOTE: comment this line out if you get errors on Mac OSX
}  # When these keys are missing from hyperparameters dict, no models of that type are trained

time_limit = 2*60  # train various models for ~2 min
num_trials = 5  # try at most 5 different hyperparameter configurations for each type of model
search_strategy = 'auto'  # to tune hyperparameters using random search routine with a local scheduler

hyperparameter_tune_kwargs = {  # HPO is not performed unless hyperparameter_tune_kwargs is specified
    'num_trials': num_trials,
    'scheduler' : 'local',
    'searcher': search_strategy,
}  # Refer to TabularPredictor.fit docstring for all valid values

predictor = TabularPredictor(label=label, eval_metric=metric).fit(
    train_data,
    time_limit=time_limit,
    hyperparameters=hyperparameters,
    hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
)
```

Use the trained models to predict on the test data:


```python
predictor.predict_proba(test_data)
```

Use leaderboard to see how each model performs on the test data:


```python
predictor.leaderboard(test_data)
```

In the above example, the predictive performance may be poor because we specified very little training to ensure quick runtimes.  You can call `fit()` multiple times while modifying the above settings to better understand how these choices affect performance outcomes. For example: you can increase `subsample_size` to train using a larger dataset, increase the `num_epochs` and `num_boost_round` hyperparameters, and increase the `time_limit` (which you should do for all code in these tutorials).  To see more detailed output during the execution of `fit()`, you can also pass in the argument: `verbosity=3`.
