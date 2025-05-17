Summary: This tutorial demonstrates how to implement custom evaluation metrics in AutoGluon using the make_scorer() function. It covers the technical implementation of creating serializable custom metrics for different types of machine learning tasks (classification, regression, probability-based) through detailed examples. The tutorial helps with tasks like defining custom accuracy, MSE, and ROC AUC metrics, integrating them into model training and evaluation workflows. Key features include the essential parameters for make_scorer(), proper metric serialization requirements, handling different prediction types (class, probability, threshold-based), and best practices for implementing custom metric functions that are compatible with AutoGluon's framework.

# Adding a custom metric to AutoGluon

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/autogluon/autogluon/blob/master/docs/tutorials/tabular/advanced/tabular-custom-metric.ipynb)
[![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/autogluon/autogluon/blob/master/docs/tutorials/tabular/advanced/tabular-custom-metric.ipynb)



**Tip**: If you are new to AutoGluon, review [Predicting Columns in a Table - Quick Start](../tabular-quick-start.ipynb) to learn the basics of the AutoGluon API.

This tutorial describes how to add a custom evaluation metric to AutoGluon that is used to inform validation scores, model ensembling, hyperparameter tuning, and more.

In this example, we show a variety of evaluation metrics and how to convert them to an AutoGluon Scorer ([Scorer source code](https://github.com/autogluon/autogluon/blob/master/core/src/autogluon/core/metrics/__init__.py)), which can then be passed to AutoGluon models and predictors.

First, we will randomly generate 10 ground truth labels and predictions and show how to calculate metric scores from them.


```python
!pip install autogluon.tabular[all]

```


```python
import numpy as np

rng = np.random.default_rng(seed=42)
y_true = rng.integers(low=0, high=2, size=10)
y_pred = rng.integers(low=0, high=2, size=10)

print(f'y_true: {y_true}')
print(f'y_pred: {y_pred}')
```

## Ensuring Metric is Serializable
Custom metrics must be defined in a separate Python file and imported so that they can be [pickled](https://docs.python.org/3/library/pickle.html) (Python's serialization protocol).
If a custom metric is not pickleable, AutoGluon will crash during fit when trying to parallelize model training with Ray.
In the below example, you would want to create a new python file such as `my_metrics.py` with `ag_accuracy_scorer` defined in it,
and then use it via `from my_metrics import ag_accuracy_scorer`.

If your metric is not serializable, you will get many errors similar to: `_pickle.PicklingError: Can't pickle`. Refer to https://github.com/autogluon/autogluon/issues/1637 for an example.
For an example of how to specify a custom metric on Kaggle, refer to [this Kaggle Notebook](https://www.kaggle.com/code/rzatemizel/prepare-for-automl-grand-prix-explore-autogluon#Custom-Metric-for-Autogluon).

The custom metrics in this tutorial are **not** serializable for ease of demonstration. If the `best_quality` preset was used, calls to `fit()` would crash.

## Custom Accuracy Metric
We will start by creating a custom accuracy metric. A prediction is correct if the predicted value is the same as the true value, otherwise it is wrong.

First, lets use the default sklearn accuracy scorer:


```python
import sklearn.metrics

sklearn.metrics.accuracy_score(y_true, y_pred)
```

There are a variety of limitations with the above logic.
For example, without outside knowledge of the metric it is unknown:
1. What the optimal value is (1)
2. If higher values are better (True)
3. If the metric requires predictions, class predictions, or class probabilities (class predictions)

Now, let's convert this evaluation metric to an AutoGluon Scorer to address these limitations.

We do this by calling `autogluon.core.metrics.make_scorer` (Source code: [autogluon/core/metrics/\_\_init\_\_.py](https://github.com/autogluon/autogluon/blob/master/core/src/autogluon/core/metrics/__init__.py)).


```python
from autogluon.core.metrics import make_scorer

ag_accuracy_scorer = make_scorer(name='accuracy',
                                 score_func=sklearn.metrics.accuracy_score,
                                 optimum=1,
                                 greater_is_better=True,
                                 needs_class=True)
```

When creating the Scorer, we need to specify a name for the Scorer. This does not need to be any particular value but is used when printing information about the Scorer during training.

Next, we specify the `score_func`. This is the function we want to wrap, in this case, sklearn's `accuracy_score` function.

We then need to specify the `optimum` value.
This is necessary when calculating `error` (also known as `regret`) as opposed to `score`.
`error` is defined as `sign * optimum - score`, where `sign=1` if `greater_is_better=True`, else `sign=-1`.
It is also useful to identify when a score is optimal and cannot be improved.
Because the best possible value from `sklearn.metrics.accuracy_score` is `1`, we specify `optimum=1`.

Next we need to specify `greater_is_better`. In this case, `greater_is_better=True`
because the best value returned is 1, and the worst value returned is less than 1 (0).
It is very important to set this value correctly,
otherwise AutoGluon will try to optimize for the **worst** model instead of the best.

Finally, we specify a bool `needs_*` based on the type of metric we are using. The following options are available: [`needs_pred`, `needs_proba`, `needs_class`, `needs_threshold`, `needs_quantile`].
All of them default to False except `needs_pred` which is inferred based on the other 4, of which only one can be set to True. If none are specified, the metric is treated as a regression metric (`needs_pred=True`).

Below is a detailed description of each:

    needs_pred : bool | str, default="auto"
        Whether score_func requires the predict model method output as input to scoring.
        If "auto", will be inferred based on the values of the other `needs_*` arguments.
        Defaults to True if all other `needs_*` are False.
        Examples: ["root_mean_squared_error", "mean_squared_error", "r2", "mean_absolute_error", "median_absolute_error", "spearmanr", "pearsonr"]

    needs_proba : bool, default=False
        Whether score_func requires predict_proba to get probability estimates out of a classifier.
        These scorers can benefit from calibration methods such as temperature scaling.
        Examples: ["log_loss", "roc_auc_ovo", "roc_auc_ovr", "pac"]

    needs_class : bool, default=False
        Whether score_func requires class predictions (classification only).
        This is required to determine if the scorer is impacted by a decision threshold.
        These scorers can benefit from decision threshold calibration methods such as via `predictor.calibrate_decision_threshold()`.
        Examples: ["accuracy", "balanced_accuracy", "f1", "precision", "recall", "mcc", "quadratic_kappa", "f1_micro", "f1_macro", "f1_weighted"]

    needs_threshold : bool, default=False
        Whether score_func takes a continuous decision certainty.
        This only works for binary classification.
        These scorers care about the rank order of the prediction probabilities to calculate their scores, and are undefined if given a single sample to score.
        Examples: ["roc_auc", "average_precision"]

    needs_quantile : bool, default=False
        Whether score_func is based on quantile predictions.
        This only works for quantile regression.
        Examples: ["pinball_loss"]

Because we are creating an accuracy scorer, we need the class prediction, and therefore we specify `needs_class=True`.

**Advanced Note**: `optimum` must correspond to the optimal value
from the original metric callable (in this case `sklearn.metrics.accuracy_score`).
Hypothetically, if a metric callable was `greater_is_better=False` with an optimal value of `-2`,
you should specify `optimum=-2, greater_is_better=False`.
In this case, if `raw_metric_value=-0.5`
then Scorer would return `score=0.5` to enforce higher_is_better (`score = sign * raw_metric_value`).
Scorer's error would be `error=1.5` because `sign (-1) * optimum (-2) - score (0.5) = 1.5`

Once created, the AutoGluon Scorer can be called in the same fashion as the original metric to compute `score`.


```python
# score
ag_accuracy_scorer(y_true, y_pred)
```

Alternatively, `.score` is an alias to the above callable for convenience:


```python
ag_accuracy_scorer.score(y_true, y_pred)
```

To get the error instead of score:


```python
# error, error=sign*optimum-score -> error=1*1-score -> error=1-score
ag_accuracy_scorer.error(y_true, y_pred)

# Can also convert score to error and vice-versa:
# score = ag_accuracy_scorer(y_true, y_pred)
# error = ag_accuracy_scorer.convert_score_to_error(score)
# score = ag_accuracy_scorer.convert_error_to_score(error)

# Can also convert score to the original score that would be returned in `score_func`:
# score_orig = ag_accuracy_scorer.convert_score_to_original(score)  # score_orig = sign * score
```

Note that `score` is in `higher_is_better` format, while error is in `lower_is_better` format.
An error of 0 corresponds to a perfect prediction.

## Custom Mean Squared Error Metric

Next, let's show examples of how to convert regression metrics into Scorers.

First we generate random ground truth labels and their predictions, however this time they are floats instead of integers.


```python
y_true = rng.random(10)
y_pred = rng.random(10)

print(f'y_true: {y_true}')
print(f'y_pred: {y_pred}')
```

A common regression metric is Mean Squared Error:


```python
sklearn.metrics.mean_squared_error(y_true, y_pred)
```


```python
ag_mean_squared_error_scorer = make_scorer(name='mean_squared_error',
                                           score_func=sklearn.metrics.mean_squared_error,
                                           optimum=0,
                                           greater_is_better=False)
```

In this case, `optimum=0` because this is an error metric.

Additionally, `greater_is_better=False` because sklearn reports error as positive values, and the lower the value is, the better.

A very important point about AutoGluon Scorers is that internally,
they will always report scores in `greater_is_better=True` form.
This means if the original metric was `greater_is_better=False`, AutoGluon's Scorer will flip the value.
Therefore, `score` will be represented as a negative value.

This is done to ensure consistency between different metrics.


```python
# score
ag_mean_squared_error_scorer(y_true, y_pred)
```


```python
# error, error=sign*optimum-score -> error=-1*0-score -> error=-score
ag_mean_squared_error_scorer.error(y_true, y_pred)
```

We can also specify metrics outside of sklearn. For example, below is a minimal implementation of mean squared error:


```python
def mse_func(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return ((y_true - y_pred) ** 2).mean()

mse_func(y_true, y_pred)
```

All that is required is that the function take two arguments: `y_true`, and `y_pred` (or `y_pred_proba`), as numpy arrays, and return a float value.

With the same code as before, we can create an AutoGluon Scorer.


```python
ag_mean_squared_error_custom_scorer = make_scorer(name='mean_squared_error',
                                                  score_func=mse_func,
                                                  optimum=0,
                                                  greater_is_better=False)
ag_mean_squared_error_custom_scorer(y_true, y_pred)
```

## Custom ROC AUC Metric

Here we show an example of a thresholding metric, `roc_auc`. A thresholding metric cares about the relative ordering of predictions, but not their absolute values.


```python
y_true = rng.integers(low=0, high=2, size=10)
y_pred_proba = rng.random(10)

print(f'y_true:       {y_true}')
print(f'y_pred_proba: {y_pred_proba}')
```


```python
sklearn.metrics.roc_auc_score(y_true, y_pred_proba)
```

We will need to specify `needs_threshold=True` in order for downstream models to properly use the metric.


```python
# Score functions that need decision values
ag_roc_auc_scorer = make_scorer(name='roc_auc',
                                score_func=sklearn.metrics.roc_auc_score,
                                optimum=1,
                                greater_is_better=True,
                                needs_threshold=True)
ag_roc_auc_scorer(y_true, y_pred_proba)
```

## Using Custom Metrics in TabularPredictor

Now that we have created several custom Scorers, let's use them for training and evaluating models.

For this tutorial, we will be using the Adult Income dataset.


```python
from autogluon.tabular import TabularDataset

train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')  # can be local CSV file as well, returns Pandas DataFrame
test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')  # another Pandas DataFrame
label = 'class'  # specifies which column we want to predict
train_data = train_data.sample(n=1000, random_state=0)  # subsample dataset for faster demo

train_data.head(5)
```


```python
from autogluon.tabular import TabularPredictor

predictor = TabularPredictor(label=label).fit(train_data, hyperparameters='toy')

predictor.leaderboard(test_data)
```

We can pass our custom metrics into `predictor.leaderboard` via the `extra_metrics` argument:


```python
predictor.leaderboard(test_data, extra_metrics=[ag_roc_auc_scorer, ag_accuracy_scorer])
```

We can also pass our custom metric into the Predictor itself by specifying it during initialization via the `eval_metric` parameter:


```python
predictor_custom = TabularPredictor(label=label, eval_metric=ag_roc_auc_scorer).fit(train_data, hyperparameters='toy')

predictor_custom.leaderboard(test_data)
```

That's all it takes to create and use custom metrics in AutoGluon!

If you create a custom metric, consider [submitting a PR](https://github.com/autogluon/autogluon/pulls) so that we can officially add it to AutoGluon!

For a tutorial on implementing custom models in AutoGluon, refer to [Adding a custom model to AutoGluon](tabular-custom-model.ipynb).

For more tutorials, refer to [Predicting Columns in a Table - Quick Start](../tabular-quick-start.ipynb) and [Predicting Columns in a Table - In Depth](../tabular-indepth.ipynb).
