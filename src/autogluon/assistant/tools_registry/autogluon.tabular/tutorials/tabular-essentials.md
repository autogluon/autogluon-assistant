Summary: This tutorial provides implementation guidance for AutoGluon's TabularPredictor, covering essential techniques for automated machine learning on tabular data. It helps with tasks including model training, prediction, evaluation, and optimization through presets. Key features include basic setup and installation, data loading without preprocessing, model training with various quality presets (best_quality to medium_quality), prediction methods (including probability predictions), model evaluation and persistence, and performance optimization techniques. The tutorial demonstrates how to handle both classification and regression tasks, configure evaluation metrics, and implement best practices for model deployment, while highlighting AutoGluon's automatic handling of feature engineering, missing data, and model ensembling.

# AutoGluon Tabular - Essential Functionality

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/autogluon/autogluon/blob/master/docs/tutorials/tabular/tabular-essentials.ipynb)
[![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/autogluon/autogluon/blob/master/docs/tutorials/tabular/tabular-essentials.ipynb)


Via a simple `fit()` call, AutoGluon can produce highly-accurate models to predict the values in one column of a data table based on the rest of the columns' values. Use AutoGluon with tabular data for both classification and regression problems. This tutorial demonstrates how to use AutoGluon to produce a classification model that predicts whether or not a person's income exceeds $50,000.

## TabularPredictor

To start, import AutoGluon's [TabularPredictor](../../api/autogluon.tabular.TabularPredictor.rst) and [TabularDataset](../../api/autogluon.core.TabularDataset.rst) classes:


```python
!pip install autogluon.tabular[all]

```


```python
from autogluon.tabular import TabularDataset, TabularPredictor
```

Load training data from a [CSV file](https://en.wikipedia.org/wiki/Comma-separated_values) into an AutoGluon Dataset object. This object is essentially equivalent to a [Pandas DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html) and the same methods can be applied to both.


```python
train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
subsample_size = 500  # subsample subset of data for faster demo, try setting this to much larger values
train_data = train_data.sample(n=subsample_size, random_state=0)
train_data.head()
```

Note that we loaded data from a CSV file stored in the cloud. You can also specify a local file-path instead if you have already downloaded the CSV file to your own machine (e.g., using [wget](https://www.gnu.org/software/wget/)).
Each row in the table `train_data` corresponds to a single training example. In this particular dataset, each row corresponds to an individual person, and the columns contain various characteristics reported during a census.

Let's first use these features to predict whether the person's income exceeds $50,000 or not, which is recorded in the `class` column of this table.


```python
label = 'class'
print(f"Unique classes: {list(train_data[label].unique())}")
```

AutoGluon works with raw data, meaning you don't need to perform any data preprocessing before fitting AutoGluon. We actively recommend that you avoid performing operations such as missing value imputation or one-hot-encoding, as AutoGluon has dedicated logic to handle these situations automatically. You can learn more about AutoGluon's preprocessing in the [Feature Engineering Tutorial](tabular-feature-engineering.ipynb).

### Training

Now we initialize and fit AutoGluon's TabularPredictor in one line of code:


```python
predictor = TabularPredictor(label=label).fit(train_data)
```

That's it! We now have a TabularPredictor that is able to make predictions on new data.

### Prediction

Next, load separate test data to demonstrate how to make predictions on new examples at inference time:


```python
test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')
test_data.head()
```

We can now use our trained models to make predictions on the new data:


```python
y_pred = predictor.predict(test_data)
y_pred.head()  # Predictions
```


```python
y_pred_proba = predictor.predict_proba(test_data)
y_pred_proba.head()  # Prediction Probabilities
```

### Evaluation

Next, we can [evaluate](../../api/autogluon.tabular.TabularPredictor.evaluate.rst) the predictor on the (labeled) test data:


```python
predictor.evaluate(test_data)
```

We can also [evaluate each model individually](../../api/autogluon.tabular.TabularPredictor.leaderboard.rst):


```python
predictor.leaderboard(test_data)
```

### Loading a Trained Predictor

Finally, we can load the predictor in a new session (or new machine) by calling [TabularPredictor.load()](../../api/autogluon.tabular.TabularPredictor.load.rst) and specifying the location of the predictor artifact on disk.


```python
predictor.path  # The path on disk where the predictor is saved
```


```python
# Load the predictor by specifying the path it is saved to on disk.
# You can control where it is saved to by setting the `path` parameter during init
predictor = TabularPredictor.load(predictor.path)
```

```{warning}

`TabularPredictor.load()` uses the `pickle` module implicitly, which is known to be insecure. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling. Never load data that could have come from an untrusted source, or that could have been tampered with. **Only load data you trust.**

```

Now you're ready to try AutoGluon on your own tabular datasets!
As long as they're stored in a popular format like CSV, you should be able to achieve strong predictive performance with just 2 lines of code:

```
from autogluon.tabular import TabularPredictor
predictor = TabularPredictor(label=<variable-name>).fit(train_data=<file-name>)
```


**Note:** This simple call to [TabularPredictor.fit()](../../api/autogluon.tabular.TabularPredictor.fit.rst) is intended for your first prototype model. In a subsequent section, we'll demonstrate how to maximize predictive performance by additionally specifying the `presets` parameter to `fit()` and the `eval_metric` parameter to `TabularPredictor()`.

## Description of fit()

Here we discuss what happened during `fit()`.

Since there are only two possible values of the `class` variable, this was a binary classification problem, for which an appropriate performance metric is _accuracy_. AutoGluon automatically infers this as well as the type of each feature (i.e., which columns contain continuous numbers vs. discrete categories). AutoGluon can also automatically handle common issues like missing data and rescaling feature values.

We did not specify separate validation data and so AutoGluon automatically chose a random training/validation split of the data. The data used for validation is separated from the training data and is used to determine the models and hyperparameter-values that produce the best results. Rather than just a single model, AutoGluon trains multiple models and ensembles them together to obtain superior predictive performance.

By default, AutoGluon tries to fit [various types of models](../../api/autogluon.tabular.models.rst) including neural networks and tree ensembles. Each type of model has various hyperparameters, which traditionally, the user would have to specify. AutoGluon automates this process.

AutoGluon automatically and iteratively tests values for hyperparameters to produce the best performance on the validation data. This involves repeatedly training models under different hyperparameter settings and evaluating their performance. This process can be computationally-intensive, so `fit()` parallelizes this process across multiple threads using [Ray](https://www.ray.io/). To control runtimes, you can specify various arguments in `fit()` such as `time_limit` as demonstrated in the subsequent **[In-Depth Tutorial](tabular-indepth.ipynb)**.

We can view what properties AutoGluon automatically inferred about our prediction task:


```python
print("AutoGluon infers problem type is: ", predictor.problem_type)
print("AutoGluon identified the following types of features:")
print(predictor.feature_metadata)
```

AutoGluon correctly recognized our prediction problem to be a **binary classification** task and decided that variables such as `age` should be represented as integers, whereas variables such as `workclass` should be represented as categorical objects. The `feature_metadata` attribute allows you to see the inferred data type of each predictive variable after preprocessing (this is its _raw_ dtype; some features may also be associated with additional _special_ dtypes if produced via feature-engineering, e.g. numerical representations of a datetime/text column).

To transform the data into AutoGluon's internal representation, we can do the following:


```python
test_data_transform = predictor.transform_features(test_data)
test_data_transform.head()
```

Notice how the data is purely numeric after pre-processing (although categorical features will still be treated as categorical downstream).

To better understand our trained predictor, we can estimate the overall importance of each feature via [TabularPredictor.feature_importance()](../../api/autogluon.tabular.TabularPredictor.feature_importance.rst):


```python
predictor.feature_importance(test_data)
```

The `importance` column is an estimate for the amount the evaluation metric score would drop if the feature were removed from the data.
Negative values of `importance` mean that it is likely to improve the results if re-fit with the feature removed.

When we call `predict()`, AutoGluon automatically predicts with the model that displayed the best performance on validation data (i.e. the weighted-ensemble).


```python
predictor.model_best
```

We can instead specify which model to use for predictions like this:

```
predictor.predict(test_data, model='LightGBM')
```

You can get the list of trained models via `.leaderboard()` or `.model_names()`:


```python
predictor.model_names()
```

The scores of predictive performance above were based on a default evaluation metric (accuracy for binary classification). Performance in certain applications may be measured by different metrics than the ones AutoGluon optimizes for by default. If you know the metric that counts in your application, you should specify it via the `eval_metric` argument as demonstrated in the next section.

## Presets

AutoGluon comes with a variety of presets that can be specified in the call to `.fit` via the `presets` argument. `medium_quality` is used by default to encourage initial prototyping, but for serious usage, the other presets should be used instead.

| Preset         | Model Quality                                          | Use Cases                                                                                                                                               | Fit Time (Ideal) | Inference Time (Relative to medium_quality) | Disk Usage |
| :------------- |:-------------------------------------------------------| :------------------------------------------------------------------------------------------------------------------------------------------------------ |:-----------------| :------------------------------------------ | :--------- |
| best_quality   | State-of-the-art (SOTA), much better than high_quality | When accuracy is what matters                                                                                                                           | 16x+             | 32x+                                        | 16x+       |
| high_quality   | Better than good_quality                               | When a very powerful, portable solution with fast inference is required: Large-scale batch inference                                                    | 16x+             | 4x                                          | 2x         |
| good_quality   | Stronger than any other AutoML Framework               | When a powerful, highly portable solution with very fast inference is required: Billion-scale batch inference, sub-100ms online-inference, edge-devices | 16x              | 2x                                          | 0.1x       |
| medium_quality | Competitive with other top AutoML Frameworks           | Initial prototyping, establishing a performance baseline                                                                                                | 1x               | 1x                                          | 1x         |

We recommend users to start with `medium_quality` to get a sense of the problem and identify any data related issues. If `medium_quality` is taking too long to train, consider subsampling the training data during this prototyping phase.  
Once you are comfortable, next try `best_quality`. Make sure to specify at least 16x the `time_limit` value as used in `medium_quality`. Once finished, you should have a very powerful solution that is often stronger than `medium_quality`.  
Make sure to consider holding out test data that AutoGluon never sees during training to ensure that the models are performing as expected in terms of performance.  
Once you evaluate both `best_quality` and `medium_quality`, check if either satisfies your needs. If neither do, consider trying `high_quality` and/or `good_quality`.  
If none of the presets satisfy requirements, refer to [Predicting Columns in a Table - In Depth](tabular-indepth.ipynb) for more advanced AutoGluon options.

## Maximizing predictive performance

**Note:** You should not call `fit()` with entirely default arguments if you are benchmarking AutoGluon-Tabular or hoping to maximize its accuracy!
To get the best predictive accuracy with AutoGluon, you should generally use it like this:


```python
time_limit = 60  # for quick demonstration only, you should set this to longest time you are willing to wait (in seconds)
metric = 'roc_auc'  # specify your evaluation metric here
predictor = TabularPredictor(label, eval_metric=metric).fit(train_data, time_limit=time_limit, presets='best_quality')
```


```python
predictor.leaderboard(test_data)
```

This command implements the following strategy to maximize accuracy:

- Specify the argument `presets='best_quality'`, which allows AutoGluon to automatically construct powerful model ensembles based on [stacking/bagging](https://arxiv.org/abs/2003.06505), and will greatly improve the resulting predictions if granted sufficient training time. The default value of `presets` is `'medium_quality'`, which produces _less_ accurate models but facilitates faster prototyping. With `presets`, you can flexibly prioritize predictive accuracy vs. training/inference speed. For example, if you care less about predictive performance and want to quickly deploy a basic model, consider using: `presets=['good_quality', 'optimize_for_deployment']`.

- Provide the parameter `eval_metric` to `TabularPredictor()` if you know what metric will be used to evaluate predictions in your application. Some other non-default metrics you might use include things like: `'f1'` (for binary classification), `'roc_auc'` (for binary classification), `'log_loss'` (for classification), `'mean_absolute_error'` (for regression), `'median_absolute_error'` (for regression). You can also define your own custom metric function. For more information refer to [Adding a custom metric to AutoGluon](advanced/tabular-custom-metric.ipynb).

- Include all your data in `train_data` and do not provide `tuning_data` (AutoGluon will split the data more intelligently to fit its needs).

- Do not specify the `hyperparameter_tune_kwargs` argument (counterintuitively, hyperparameter tuning is not the best way to spend a limited training time budgets, as model ensembling is often superior). We recommend you only use `hyperparameter_tune_kwargs` if your goal is to deploy a single model rather than an ensemble.

- Do not specify the `hyperparameters` argument (allow AutoGluon to adaptively select which models/hyperparameters to use).

- Set `time_limit` to the longest amount of time (in seconds) that you are willing to wait. AutoGluon's predictive performance improves the longer `fit()` is allowed to run.

## Regression (predicting numeric table columns):

To demonstrate that `fit()` can also automatically handle regression tasks, we now try to predict the numeric `age` variable in the same table based on the other features:


```python
age_column = 'age'
train_data[age_column].head()
```

We again call `fit()`, imposing a time-limit this time (in seconds), and also demonstrate a shorthand method to evaluate the resulting model on the test data (which contain labels):


```python
predictor_age = TabularPredictor(label=age_column, path="agModels-predictAge").fit(train_data, time_limit=60)
```


```python
predictor_age.evaluate(test_data)
```

Note that we didn't need to tell AutoGluon this is a regression problem, it automatically inferred this from the data and reported the appropriate performance metric (RMSE by default). To specify a particular evaluation metric other than the default, set the `eval_metric` parameter of [TabularPredictor()](../../api/autogluon.tabular.TabularPredictor.rst) and AutoGluon will tailor its models to optimize your metric (e.g. `eval_metric = 'mean_absolute_error'`). For evaluation metrics where higher values are worse (like RMSE), AutoGluon will flip their sign and print them as negative values during training (as it internally assumes higher values are better). You can even specify a custom metric by following the [Custom Metric Tutorial](advanced/tabular-custom-metric.ipynb).

We can call leaderboard to see the per-model performance:


```python
predictor_age.leaderboard(test_data)
```

**Data Formats:** AutoGluon can currently operate on data tables already loaded into Python as pandas DataFrames, or those stored in files of [CSV format](https://en.wikipedia.org/wiki/Comma-separated_values) or [Parquet format](https://databricks.com/glossary/what-is-parquet). If your data lives in multiple tables, you will first need to join them into a single table whose rows correspond to statistically independent observations (datapoints) and columns correspond to different features (aka. variables/covariates).

Refer to the [TabularPredictor documentation](../../api/autogluon.tabular.TabularPredictor.rst) to see all of the available methods/options.

## Advanced Usage

For more advanced usage examples of AutoGluon, refer to the [In Depth Tutorial](tabular-indepth.ipynb)

If you are interested in deployment optimization, refer to the [Deployment Optimization Tutorial](advanced/tabular-deployment.ipynb).

For adding custom models to AutoGluon, refer to the [Custom Model](advanced/tabular-custom-model.ipynb) and [Custom Model Advanced](advanced/tabular-custom-model-advanced.ipynb) tutorials.
