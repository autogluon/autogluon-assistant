Summary: This tutorial covers advanced AutoGluon Tabular techniques for production ML systems, including model ensembling (bagging with `num_bag_folds`, stacking with `num_stack_levels`), binary classification decision threshold calibration for metric optimization, comprehensive model evaluation via leaderboards with custom metrics, and feature importance analysis. It provides detailed inference acceleration strategies prioritized as: `refit_full()` for collapsing bagged ensembles (8x speedup), `persist()` for keeping models in memory (10x speedup in online inference), and `infer_limit` constraints during training (50x+ speedup). Additional optimization techniques include Pareto-optimal weighted ensembles, model distillation, preprocessing caching, and memory/disk management strategies. The tutorial helps with tasks like deploying fast prediction systems, optimizing model accuracy-latency tradeoffs, reducing resource usage, cross-machine model deployment, and implementing interpretable ML solutions with proper metric calibration.

```python
!pip install autogluon.tabular[all]

```


```python
from autogluon.tabular import TabularDataset, TabularPredictor

train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
subsample_size = 1000  # subsample subset of data for faster demo, try setting this to much larger values
train_data = train_data.sample(n=subsample_size, random_state=0)
test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')

label = 'class'  # Now lets predict the "class" column (binary classification)
metric = "balanced_accuracy"  # Specify the evaluation metric you want to optimize
X_test = test_data.drop(columns=[label])
y_test = test_data[label]

train_data.head()
```

## Model ensembling with stacking/bagging

Two methods to boost predictive performance are [bagging and stack-ensembling](https://arxiv.org/abs/2003.06505).  You'll often see performance improve if you specify `num_bag_folds` = 8, `num_stack_levels` = 1 in the call to `fit()`, but this will increase training times and memory/disk usage.


```python
save_path = 'agModels-predictClass'  # folder where to store trained models

predictor = TabularPredictor(label=label, eval_metric=metric, path=save_path).fit(train_data,
    calibrate_decision_threshold=False,  # Disabling for demonstration in next section
    num_bag_folds=8, num_bag_sets=1, num_stack_levels=1,  # or simply set `auto_stack=True` or a preset such as `presets="best"`
    time_limit=180,  # a brief 3-minute time limit for demonstration
)
predictor.leaderboard(test_data)
```

You should not provide `tuning_data` when stacking/bagging, and instead provide all your available data as `train_data` (which AutoGluon will split in more intellgent ways). `num_bag_sets` controls how many times the k-fold bagging process is repeated to further reduce variance (increasing this may further boost accuracy but will substantially increase training times, inference latency, and memory/disk usage). Rather than manually searching for good bagging/stacking values yourself, AutoGluon will automatically select good values for you if you specify `auto_stack` instead (which is used in the `best_quality` preset):

## Decision Threshold Calibration

Major metric score improvements can be achieved in binary classification for metrics such as `"f1"` and `"balanced_accuracy"` by adjusting the prediction decision threshold via `calibrate_decision_threshold` to a value other than the default 0.5.

Below is an example of the `"balanced_accuracy"` score achieved on the test data with and without calibrating the decision threshold:


```python
print(f'Prior to calibration (predictor.decision_threshold={predictor.decision_threshold}):')
scores = predictor.evaluate(test_data)

calibrated_decision_threshold = predictor.calibrate_decision_threshold()
predictor.set_decision_threshold(calibrated_decision_threshold)

print(f'After calibration (predictor.decision_threshold={predictor.decision_threshold}):')
scores_calibrated = predictor.evaluate(test_data)
```


```python
for metric_name in scores:
    metric_score = scores[metric_name]
    metric_score_calibrated = scores_calibrated[metric_name]
    decision_threshold = predictor.decision_threshold
    print(f'decision_threshold={decision_threshold:.3f}\t| metric="{metric_name}"'
          f'\n\ttest_score uncalibrated: {metric_score:.4f}'
          f'\n\ttest_score   calibrated: {metric_score_calibrated:.4f}'
          f'\n\ttest_score        delta: {metric_score_calibrated-metric_score:.4f}')
```

Notice that calibrating for "balanced_accuracy" majorly improved the "balanced_accuracy" metric score, but it harmed the "accuracy" score. Threshold calibration will often result in a tradeoff between performance on different metrics, and the user should keep this in mind.

Instead of calibrating for "balanced_accuracy" specifically, we can calibrate for any metric if we want to maximize the score of that metric:


```python
predictor.set_decision_threshold(0.5)  # Reset decision threshold
for metric_name in ['f1', 'balanced_accuracy', 'mcc']:
    metric_score = predictor.evaluate(test_data, silent=True)[metric_name]
    calibrated_decision_threshold = predictor.calibrate_decision_threshold(metric=metric_name, verbose=False)
    metric_score_calibrated = predictor.evaluate(
        test_data, decision_threshold=calibrated_decision_threshold, silent=True
    )[metric_name]
    print(f'decision_threshold={calibrated_decision_threshold:.3f}\t| metric="{metric_name}"'
          f'\n\ttest_score uncalibrated: {metric_score:.4f}'
          f'\n\ttest_score   calibrated: {metric_score_calibrated:.4f}'
          f'\n\ttest_score        delta: {metric_score_calibrated-metric_score:.4f}')
```

Instead of calibrating the decision threshold post-fit, you can have it automatically occur during the fit call by specifying the fit parameter `predictor.fit(..., calibrate_decision_threshold=True)`.

Luckily, AutoGluon will automatically apply decision threshold calibration when beneficial, as the default value is `calibrate_decision_threshold="auto"`. We recommend keeping this value as the default in most cases.

Additional usage examples are below:

```python
# Will use the decision_threshold specified in `predictor.decision_threshold`, can be set via `predictor.set_decision_threshold`
y_pred = predictor.predict(test_data)
y_pred_08 = predictor.predict(test_data, decision_threshold=0.8)  # Specify a specific threshold to use only for this predict

y_pred_proba = predictor.predict_proba(test_data)
y_pred = predictor.predict_from_proba(y_pred_proba)  # Identical output to calling .predict(test_data)
y_pred_08 = predictor.predict_from_proba(y_pred_proba, decision_threshold=0.8)  # Identical output to calling .predict(test_data, decision_threshold=0.8)
```

## Prediction options (inference)

Even if you've started a new Python session since last calling `fit()`, you can still load a previously trained predictor from disk:


```python
predictor = TabularPredictor.load(save_path)  # `predictor.path` is another way to get the relative path needed to later load the predictor.
```

Above `save_path` is the same folder previously passed to `TabularPredictor`, in which all the trained models have been saved. You can train easily models on one machine and deploy them on another. Simply copy the `save_path` folder to the new machine and specify its new path in `TabularPredictor.load()`.

To find out the required feature columns to make predictions, call `predictor.features()`:


```python
predictor.features()
```

We can make a prediction on an individual example rather than a full dataset:


```python
datapoint = X_test.iloc[[0]]  # Note: .iloc[0] won't work because it returns pandas Series instead of DataFrame
datapoint
```


```python
predictor.predict(datapoint)
```

To output predicted class probabilities instead of predicted classes, you can use:


```python
predictor.predict_proba(datapoint)  # returns a DataFrame that shows which probability corresponds to which class
```

By default, `predict()` and `predict_proba()` will utilize the model that AutoGluon thinks is most accurate, which is usually an ensemble of many individual models. Here's how to see which model this is:


```python
predictor.model_best
```

We can instead specify a particular model to use for predictions (e.g. to reduce inference latency). Note that a 'model' in AutoGluon may refer to, for example, a single Neural Network, a bagged ensemble of many Neural Network copies trained on different training/validation splits, a weighted ensemble that aggregates the predictions of many other models, or a stacker model that operates on predictions output by other models. This is akin to viewing a Random Forest as one 'model' when it is in fact an ensemble of many decision trees.


Before deciding which model to use, let's evaluate all of the models AutoGluon has previously trained on our test data:


```python
predictor.leaderboard(test_data)
```

The leaderboard shows each model's predictive performance on the test data (`score_test`) and validation data (`score_val`), as well as the time required to: produce predictions for the test data (`pred_time_val`), produce predictions on the validation data (`pred_time_val`), and train only this model (`fit_time`). Below, we show that a leaderboard can be produced without new data (just uses the data previously reserved for validation inside `fit`) and can display extra information about each model:


```python
predictor.leaderboard(extra_info=True)
```

The expanded leaderboard shows properties like how many features are used by each model (`num_features`), which other models are ancestors whose predictions are required inputs for each model (`ancestors`), and how much memory each model and all its ancestors would occupy if simultaneously persisted (`memory_size_w_ancestors`). See the [leaderboard documentation](../../api/autogluon.tabular.TabularPredictor.leaderboard.rst) for full details.

To show scores for other metrics, you can specify the `extra_metrics` argument when passing in `test_data`:


```python
predictor.leaderboard(test_data, extra_metrics=['accuracy', 'balanced_accuracy', 'log_loss'])
```

Notice that `log_loss` scores are negative.
This is because metrics in AutoGluon are always shown in `higher_is_better` form.
This means that metrics such as `log_loss` and `root_mean_squared_error` will have their signs FLIPPED, and values will be negative.
This is necessary to avoid the user needing to know the metric to understand if higher is better when looking at leaderboard.

Here's how to specify a particular model to use for prediction instead of AutoGluon's default model-choice:


```python
i = 0  # index of model to use
all_models = predictor.model_names()
model_to_use = all_models[i]
model_pred = predictor.predict(datapoint, model=model_to_use)
print("Prediction from %s model: %s" % (model_to_use, model_pred.iloc[0]))
```

We can easily access various information about the trained predictor or a particular model:


```python
# Objects defined below are dicts of various information (not printed here as they are quite large):
predictor_information = predictor.info()  # access info about the predictor
model_info = predictor.model_info(model_to_use)  # access info about a model
model_info_alternative = predictor._trainer.load_model(model_to_use).get_info()  # load the inner model and access its info directly
```

The `predictor` also remembers what metric predictions should be evaluated with, which can be done with ground truth labels as follows:


```python
y_pred_proba = predictor.predict_proba(X_test)
predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred_proba)
```

Since the label columns remains in the `test_data` DataFrame, we can instead use the shorthand:


```python
predictor.evaluate(test_data)
```

## Interpretability (feature importance)

To better understand our trained predictor, we can estimate the overall importance of each feature:


```python
predictor.feature_importance(test_data)
```

Computed via [permutation-shuffling](https://explained.ai/rf-importance/), these feature importance scores quantify the drop in predictive performance (of the already trained predictor) when one column's values are randomly shuffled across rows. The top features in this list contribute most to AutoGluon's accuracy (for predicting when/if a patient will be readmitted to the hospital). Features with non-positive importance score hardly contribute to the predictor's accuracy, or may even be actively harmful to include in the data (consider removing these features from your data and calling `fit` again). These scores facilitate interpretability of the predictor's global behavior (which features it relies on for *all* predictions).
To get [local explanations](https://christophm.github.io/interpretable-ml-book/taxonomy-of-interpretability-methods.html) regarding which features influence a *particular* prediction, check out the [example notebooks](https://github.com/autogluon/autogluon/tree/master/examples/tabular/interpret) for explaining particular AutoGluon predictions using [Shapely values](https://github.com/slundberg/shap/).

Before making judgement on if AutoGluon is more or less interpretable than another solution, we recommend reading [The Mythos of Model Interpretability](https://dl.acm.org/doi/pdf/10.1145/3236386.3241340) by Zachary Lipton, which covers why often-claimed interpretable models such as trees and linear models are rarely meaningfully more interpretable than more advanced models.

## Accelerating inference

We describe multiple ways to reduce the time it takes for AutoGluon to produce predictions.

Before providing code examples, it is important to understand that
there are several ways to accelerate inference in AutoGluon. The table below lists the options in order of priority.

| Optimization                      | Inference Speedup                                                                                   | Cost                           | Notes                                                                                                                                                        |
|:----------------------------------|:----------------------------------------------------------------------------------------------------|:-------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| refit_full                        | 8x (requires bagging)                                                                               | -Quality                       | Only provides speedup with bagging enabled.                                                                                                                  |
| persist                           | Up to 10x in online-inference                                                                       | +Memory Usage                  | If memory is not sufficient to persist model, speedup is not gained. Speedup is most effective in online-inference and is not relevant in batch inference.   |
| infer_limit                       | Up to 50x+                                                                                          | -Quality (Relative to speedup) | Best when combined with refit_full.                                                                                                                          |
| feature pruning                   | Typically at most 1.5x. More if willing to lower quality significantly.                             | -Quality?                      | Dependent on the existence of unimportant features in data. Call `predictor.feature_importance(test_data)` to gauge which features could be removed.         |
| use faster hardware               | Usually at most 3x. Depends on hardware (ignoring GPU).                                             | +Hardware                      | As an example, an EC2 c6i.2xlarge is ~1.6x faster than an m5.2xlarge for a similar price. Laptops in particular might be slow compared to cloud instances.   |
| manual hyperparameters adjustment | Usually at most 2x assuming infer_limit is already specified.                                       | Manual Effort                  | Can be very complicated and is not recommended. Potential ways to get speedups this way is to reduce the number of trees in LightGBM, XGBoost, and CatBoost. |
| manual data preprocessing         | Usually at most 1.2x.                                                                               | Manual Effort                  | Only relevant for online-inference. Not recommended as AutoGluon's default preprocessing is highly optimized.                                                |

For most scenarios, the order of inference optimizations should be:
1. refit_full  
2. persist  
3. infer_limit

If following these recommendations does not lead to a sufficiently fast model, you may consider the more advanced options in the table.

### persist: Keeping models in memory

By default, AutoGluon loads models into memory one at a time and only when they are needed for prediction. This strategy is robust for large stacked/bagged ensembles, but leads to slower prediction times. If you plan to repeatedly make predictions (e.g. on new datapoints one at a time rather than one large test dataset), you can first specify that all models required for inference should be loaded into memory as follows:


```python
import time
import numpy as np

num_test = 20
preds = np.array(['']*num_test, dtype='object')
time_start = time.time()
for i in range(num_test):
    datapoint = X_test.iloc[[i]]
    pred_numpy = predictor.predict(datapoint, as_pandas=False)
    preds[i] = pred_numpy[0]
time_end = time.time()
time_without_persist = (time_end - time_start) / num_test
```


```python
predictor.persist()

preds = np.array(['']*num_test, dtype='object')
time_start = time.time()
for i in range(num_test):
    datapoint = X_test.iloc[[i]]
    pred_numpy = predictor.predict(datapoint, as_pandas=False)
    preds[i] = pred_numpy[0]
time_end = time.time()
time_with_persist = (time_end - time_start) / num_test

predictor.unpersist()  # free memory by clearing models, future predict() calls will load models from disk
```


```python
print(f"Inference time unoptimized:  {time_without_persist:.3f}s")
print(f"Inference time with persist: {time_with_persist:.3f}s")
```

You can alternatively specify a particular model to persist via the `models` argument of `persist()`, or simply set `models='all'` to simultaneously load every single model that was trained during `fit`.

### infer_limit: Inference speed as a fit constraint

If you know your latency constraint prior to fitting the predictor, you can specify it explicitly as a fit argument.
AutoGluon will then automatically train models in a fashion that attempts to satisfy the constraint.

This constraint has two components: `infer_limit` and `infer_limit_batch_size`:  
- `infer_limit` is the time in seconds to predict 1 row of data.
For example, `infer_limit=0.05` means 50 ms per row of data,
or 20 rows / second throughput.  
- `infer_limit_batch_size` is the amount of rows passed at once to predict when calculating per-row speed.
This is very important because `infer_limit_batch_size=1` (online-inference) is highly suboptimal as
various operations have a fixed cost overhead regardless of data size. If you can pass your test data in bulk,
you should specify `infer_limit_batch_size=10000`.


```python
# At most 0.05 ms per row (20000 rows per second throughput)
infer_limit = 0.00005
# adhere to infer_limit with batches of size 10000 (batch-inference, easier to satisfy infer_limit)
infer_limit_batch_size = 10000
# adhere to infer_limit with batches of size 1 (online-inference, much harder to satisfy infer_limit)
# infer_limit_batch_size = 1  # Note that infer_limit<0.02 when infer_limit_batch_size=1 can be difficult to satisfy.
predictor_infer_limit = TabularPredictor(label=label, eval_metric=metric).fit(
    train_data=train_data,
    time_limit=30,
    infer_limit=infer_limit,
    infer_limit_batch_size=infer_limit_batch_size,
)

# NOTE: If bagging was enabled, it is important to call refit_full at this stage.
#  infer_limit assumes that the user will call refit_full after fit.
# predictor_infer_limit.refit_full()

# NOTE: To align with inference speed calculated during fit, models must be persisted.
predictor_infer_limit.persist()
# Below is an optimized version that only persists the minimum required models for prediction.
# predictor_infer_limit.persist('best')

predictor_infer_limit.leaderboard()
```

Now we can test the inference speed of the final model and check if it satisfies the inference constraints.


```python
test_data_batch = test_data.sample(infer_limit_batch_size, replace=True, ignore_index=True)

import time
time_start = time.time()
predictor_infer_limit.predict(test_data_batch)
time_end = time.time()

infer_time_per_row = (time_end - time_start) / len(test_data_batch)
rows_per_second = 1 / infer_time_per_row
infer_time_per_row_ratio = infer_time_per_row / infer_limit
is_constraint_satisfied = infer_time_per_row_ratio <= 1

print(f'Model is able to predict {round(rows_per_second, 1)} rows per second. (User-specified Throughput = {1 / infer_limit})')
print(f'Model uses {round(infer_time_per_row_ratio * 100, 1)}% of infer_limit time per row.')
print(f'Model satisfies inference constraint: {is_constraint_satisfied}')
```

### Using smaller ensembles for prediction

Without having to retrain any models, one can construct alternative ensembles that aggregate individual models' predictions with different weighting schemes. These ensembles become smaller (and hence faster for prediction) if they assign nonzero weight to less models. You can produce a wide variety of ensembles with different accuracy-speed tradeoffs like this:


```python
additional_ensembles = predictor.fit_weighted_ensemble(expand_pareto_frontier=True)
print("Alternative ensembles you can use for prediction:", additional_ensembles)

predictor.leaderboard(only_pareto_frontier=True)
```

The resulting leaderboard will contain the most accurate model for a given inference-latency. You can select whichever model exhibits acceptable latency from the leaderboard and use it for prediction.


```python
model_for_prediction = additional_ensembles[0]
predictions = predictor.predict(test_data, model=model_for_prediction)
predictor.delete_models(models_to_delete=additional_ensembles)  # delete these extra models so they don't affect rest of tutorial
```

### Collapsing bagged ensembles via refit_full

For an ensemble predictor trained with bagging (as done above), recall there are ~10 bagged copies of each individual model trained on different train/validation folds. We can collapse this bag of ~10 models into a single model that's fit to the full dataset, which can greatly reduce its memory/latency requirements (but may also reduce accuracy). Below we refit such a model for each original model but you can alternatively do this for just a particular model by specifying the `model` argument of `refit_full()`.


```python
refit_model_map = predictor.refit_full()
print("Name of each refit-full model corresponding to a previous bagged ensemble:")
print(refit_model_map)
predictor.leaderboard(test_data)
```

This adds the refit-full models to the leaderboard and we can opt to use any of them for prediction just like any other model. Note `pred_time_test` and `pred_time_val` list the time taken to produce predictions with each model (in seconds) on the test/validation data. Since the refit-full models were trained using all of the data, there is no internal validation score (`score_val`) available for them. You can also call `refit_full()` with non-bagged models to refit the same models to your full dataset (there won't be memory/latency gains in this case but test accuracy may improve).

### Model distillation

While computationally-favorable, single individual models will usually have lower accuracy than weighted/stacked/bagged ensembles. [Model Distillation](https://arxiv.org/abs/2006.14284) offers one way to retain the computational benefits of a single model, while enjoying some of the accuracy-boost that comes with ensembling. The idea is to train the individual model (which we can call the student) to mimic the predictions of the full stack ensemble (the teacher). Like `refit_full()`, the `distill()` function will produce additional models we can opt to use for prediction.


```python
student_models = predictor.distill(time_limit=120)
predictor.leaderboard(test_data)
```

While distillation might produce efficient models, we recommend first focusing on `refit_full`, `infer_limit`, and `persist` to try satisfying your requirements, as distillation may come with a significant accuracy drop.

### Faster presets or hyperparameters

Instead of trying to speed up a cumbersome trained model at prediction time, if you know inference latency or memory will be an issue at the outset, then you can adjust the training process accordingly to ensure `fit()` does not produce unwieldy models.

One option is to specify more lightweight `presets`:

```python
presets = ['good_quality', 'optimize_for_deployment']
predictor_light = TabularPredictor(label=label, eval_metric=metric).fit(train_data, presets=presets, time_limit=30)
```

You may also exclude specific unwieldy models from being trained at all. Below we exclude models that tend to be slower (K Nearest Neighbors, Neural Networks):

```python
excluded_model_types = ['KNN', 'NN_TORCH']
predictor_light = TabularPredictor(label=label, eval_metric=metric).fit(train_data, excluded_model_types=excluded_model_types, time_limit=30)
```

### (Advanced) Cache preprocessed data

If you are repeatedly predicting on the same data you can cache the preprocessed version of the data and
directly send the preprocessed data to `predictor.predict` for faster inference:

```
test_data_preprocessed = predictor.transform_features(test_data)

# The following call will be faster than a normal predict call because we are skipping the preprocessing stage.
predictions = predictor.predict(test_data_preprocessed, transform_features=False)
```


Note that this is only useful in situations where you are repeatedly predicting on the same data.
If this significantly speeds up your use-case, consider whether your current approach makes sense
or if a cache on the predictions is a better solution. 

### (Advanced) Disable preprocessing

If you would rather do data preprocessing outside of TabularPredictor,
you can disable TabularPredictor's preprocessing entirely via:

```
predictor.fit(..., feature_generator=None, feature_metadata=YOUR_CUSTOM_FEATURE_METADATA)
```


Be warned that this removes ALL guardrails on data sanitization.
It is very likely that you will run into errors doing this unless you are very familiar with AutoGluon.

One instance where this can be helpful is if you have many problems
that re-use the exact same data with the exact same features. If you had 30 tasks that re-use the same features,
you could fit a `autogluon.features` feature generator once on the data, and then when you need to
predict on the 30 tasks, preprocess the data only once and then send the preprocessed data to all 30 predictors.


## If you encounter memory issues

To reduce memory usage during training, you may try each of the following strategies individually or combinations of them (these may harm accuracy):

- In `fit()`, set `excluded_model_types = ['KNN', 'XT' ,'RF']` (or some subset of these models).
- Try different `presets` in `fit()`.
- Text fields in your table require substantial memory for N-gram featurization. To mitigate this in `fit()`, you can either: (1) add `'ignore_text'` to your `presets` list (to ignore text features), or (2) specify the argument:

```python
from sklearn.feature_extraction.text import CountVectorizer
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
MAX_NGRAM = 1000
feature_generator = AutoMLPipelineFeatureGenerator(vectorizer=CountVectorizer(min_df=30, ngram_range=(1, 3), max_features=MAX_NGRAM, dtype=np.uint8))
predictor = TabularPredictor(...).fit(..., feature_generator=feature_generator)
```

For example using `MAX_NGRAM = 1000` (try various values under 10000 to reduce the number of N-gram features used to represent each text field)

In addition to reducing memory usage, many of the above strategies can also be used to reduce training times.

To reduce memory usage during inference:

- If trying to produce predictions for a large test dataset, break the test data into smaller chunks as demonstrated in [FAQ](tabular-faq.ipynb).

- If models have been previously persisted in memory but inference-speed is not a major concern, call `predictor.unpersist()`.

- If models have been previously persisted in memory, bagging was used in `fit()`, and inference-speed is a concern: call `predictor.refit_full()` and use one of the refit-full models for prediction (ensure this is the only model persisted in memory).



## If you encounter disk space issues

To reduce disk usage, you may try each of the following strategies individually or combinations of them:

- Make sure to delete all `predictor.path` folders from previous `fit()` runs! These can eat up your free space if you call `fit()` many times. If you didn't specify `path`, AutoGluon still automatically saved its models to a folder called: "AutogluonModels/ag-[TIMESTAMP]", where TIMESTAMP records when `fit()` was called, so make sure to also delete these folders if you run low on free space.

- Call `predictor.save_space()` to delete auxiliary files produced during `fit()`.

- Call `predictor.delete_models(models_to_keep='best')` if you only intend to use this predictor for inference going forward (will delete files required for non-prediction-related functionality like `fit_summary`).

- In `fit()`, you can add `'optimize_for_deployment'` to the `presets` list, which will automatically invoke the previous two strategies after training.

- Most of the above strategies to reduce memory usage will also reduce disk usage (but may harm accuracy).


## References

The following paper describes how AutoGluon internally operates on tabular data:

Erickson et al. [AutoGluon-Tabular: Robust and Accurate AutoML for Structured Data](https://arxiv.org/abs/2003.06505). *Arxiv*, 2020.

## Next Steps

If you are interested in deployment optimization, refer to the [Predicting Columns in a Table - Deployment Optimization](advanced/tabular-deployment.ipynb) tutorial.
