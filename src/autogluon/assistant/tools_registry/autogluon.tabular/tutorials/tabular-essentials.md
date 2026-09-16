Summary: This tutorial covers AutoGluon's TabularPredictor for automated machine learning on tabular data, including classification and regression tasks. It demonstrates minimal 2-line training code, automatic problem type detection, feature engineering, model ensembling, and hyperparameter tuning. Key implementations include data loading with TabularDataset (CSV/Parquet support), prediction methods (predict/predict_proba), model evaluation (evaluate/leaderboard), feature importance analysis, and model persistence. Critical configurations covered: `presets` parameter ('extreme' for GPU, 'best' for CPU production), `eval_metric` specification (f1, roc_auc, RMSE, custom metrics), `time_limit` for training duration, and best practices like avoiding manual preprocessing, using raw data, and not specifying hyperparameters manually. The tutorial helps with building production ML pipelines, optimizing model performance, selecting appropriate presets based on hardware/requirements, and understanding AutoGluon's automatic feature handling and ensemble strategies.

```python
!pip install autogluon.tabular[all]

```


```python
from autogluon.tabular import TabularDataset, TabularPredictor
```

Load training data from a [CSV file](https://en.wikipedia.org/wiki/Comma-separated_values) using AutoGluon's TabularDataset. TabularDataset is a convenience wrapper around a [Pandas DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html) and the same methods can be applied to both.


```python
train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')  # returns a pandas DataFrame, also works with parquet files
subsample_size = 1000  # subsample data for a faster demo
train_data = train_data.sample(n=subsample_size, random_state=0)
train_data.head()
```

Note that we loaded data from a CSV file stored in the cloud. You can also specify a local file instead to try AutoGluon on your own data.
Each row in the table `train_data` corresponds to a single training example. In this particular dataset, each row corresponds to an individual person, and the columns contain various characteristics reported during a census.

Let's use these features to predict whether a person's income exceeds $50,000 or not, indicated by the `class` column.


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

Next, load test data to make predictions on new examples:


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
Achieve strong predictive performance with just 2 lines of code:

```python
from autogluon.tabular import TabularPredictor
predictor = TabularPredictor(label="your_label_name").fit(train_data="train_data.csv")
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

```python
predictor.predict(test_data, model='LightGBM')
```

You can get the list of trained models via `.leaderboard()` or `.model_names()`:


```python
predictor.model_names()
```

The scores of predictive performance above were based on a default evaluation metric (accuracy for binary classification). Performance in certain applications may be measured by different metrics than the ones AutoGluon optimizes for by default. If you know the metric that counts in your application, you should specify it via the `eval_metric` argument as demonstrated in the next section.

## Presets

AutoGluon comes with a variety of presets that can be specified in the call to `.fit` via the `presets` argument. `medium` is used by default to encourage initial prototyping, but for serious usage, the other presets should be used instead.

| Preset  | Model Quality                                               | Use Cases                                                                                                                                                                                          | Fit Time (Ideal) | Inference Time (Relative to medium_quality) | Disk Usage |
|:--------|:------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------| :------------------------------------------ |:-----------|
| extreme | **Far better** than best on datasets <100000 samples | (New in v1.6) The absolute cutting edge. Incorporates recent tabular foundation models Nori, TabICLv2, and TabDPT-Turbo. Every model is free for commercial use. Requires a GPU for best results. | 1x               | 8x                                          | 2x         |
| noncommercial | **Far better** than best on datasets <100000 samples | (New in v1.6) `extreme` plus TabPFN-3, a frontier tabular foundation model created by Prior Labs. Commercial use of TabPFN-3 requires a license from Prior Labs ([license FAQ](https://docs.priorlabs.ai/models#tabpfn-model-license)). Requires a GPU for best results. | 1x               | 8x                                          | 2x         |
| best    | State-of-the-art (SOTA), much better than high      | When accuracy is what matters.  This should be considered the preferred setting for serious usage. Has been used to win numerous Kaggle competitions.                                              | 16x+             | 32x+                                        | 16x+       |
| high    | Better than good                                    | When a very powerful, portable solution with fast inference is required: Large-scale batch inference                                                                                               | 16x+             | 4x                                          | 2x         |
| good    | Stronger than any other AutoML Framework                    | When a powerful, highly portable solution with very fast inference is required: Billion-scale batch inference, sub-100ms online-inference, edge-devices                                            | 16x              | 2x                                          | 0.1x       |
| medium  | Competitive with other top AutoML Frameworks                | Initial prototyping, establishing a performance baseline                                                                                                                                           | 1x               | 1x                                          | 1x         |

**If you have a GPU, start with `extreme`.** It is meta-learned from TabArena (https://tabarena.ai) and is far better than `best` on datasets below 100000 samples, while training faster and producing a smaller predictor. Install its dependencies with `pip install autogluon[tabarena]`. `noncommercial` adds TabPFN-3 on top of the same portfolio and is stronger again; it is free for research and internal experimentation, and commercial use requires a license from Prior Labs ([license FAQ](https://docs.priorlabs.ai/models#tabpfn-model-license)). Every model in `extreme` itself is free for commercial use.

**Without a GPU, start with `best`** to get a strong performance baseline. If `best` is taking too long to train, consider running `medium` or subsampling the training data during this prototyping phase.
Make sure to consider holding out test data that AutoGluon never sees during training to ensure that the models are performing as expected in terms of performance.  
Once you evaluate both `best` and `medium`, check if either satisfies your needs. If neither do, consider trying `high` and/or `good`.  

If none of the presets satisfy requirements, refer to [Predicting Columns in a Table - In Depth](tabular-indepth.ipynb) for more advanced AutoGluon options.

## Maximizing predictive performance

**Note:** You should not call `fit()` with entirely default arguments if you are benchmarking AutoGluon-Tabular or hoping to maximize its accuracy!
To get the best predictive accuracy with AutoGluon, you should generally use it like this:


```python
time_limit = 60  # for quick demonstration only, you should set this to longest time you are willing to wait (in seconds)
metric = 'roc_auc'  # specify your evaluation metric here
predictor = TabularPredictor(label, eval_metric=metric).fit(train_data, time_limit=time_limit, presets='best')
```


```python
predictor.leaderboard(test_data)
```

This command implements the following strategy to maximize accuracy:

- Specify the argument `presets='best'`, which allows AutoGluon to automatically construct powerful model ensembles based on [stacking/bagging](https://arxiv.org/abs/2003.06505), and will greatly improve the resulting predictions if granted sufficient training time. The default value of `presets` is `'medium'`, which produces _less_ accurate models but facilitates faster prototyping. With `presets`, you can flexibly prioritize predictive accuracy vs. training/inference speed. For example, if you care less about predictive performance and want to quickly deploy a basic model, consider using: `presets=['good', 'optimize_for_deployment']`.

- Provide the parameter `eval_metric` to `TabularPredictor()` if you know what metric will be used to evaluate predictions in your application. Some other non-default metrics you might use include things like: `'f1'` (for binary classification), `'roc_auc'` (for binary classification), `'log_loss'` (for classification), `'mean_absolute_error'` (for regression), `'median_absolute_error'` (for regression). You can also define your own custom metric function. For more information refer to [Adding a custom metric to AutoGluon](advanced/tabular-custom-metric.ipynb).

- Include all your data in `train_data` and do not provide `tuning_data` (AutoGluon will split the data more intelligently to fit its needs).

- Do not specify the `hyperparameter_tune_kwargs` argument (counterintuitively, hyperparameter tuning is not the best way to spend a limited training time budgets, as model ensembling is often superior). We recommend you only use `hyperparameter_tune_kwargs` if your goal is to deploy a single model rather than an ensemble.

- Do not specify the `hyperparameters` argument (allow AutoGluon to adaptively select which models/hyperparameters to use).

- Set `time_limit` to the longest amount of time (in seconds) that you are willing to wait. AutoGluon's predictive performance improves the longer `fit()` is allowed to run.

## Regression (predicting numeric table columns):

To demonstrate that `fit()` can also automatically handle regression tasks, we now train to predict the numeric `age` variable in the same table based on the other features:


```python
age_column = 'age'
predictor_age = TabularPredictor(label=age_column, path="agModels-predictAge").fit(train_data, time_limit=30)
```


```python
predictor_age.leaderboard(test_data)
```

Note that we didn't need to tell AutoGluon this is a regression problem, it automatically inferred this from the data and used an appropriate evaluation metric (RMSE by default). To specify a particular evaluation metric other than the default, set the `eval_metric` parameter of [TabularPredictor()](../../api/autogluon.tabular.TabularPredictor.rst) and AutoGluon will tailor its models to optimize your metric (e.g. `eval_metric = 'mean_absolute_error'`). For evaluation metrics where higher values are worse (like RMSE), AutoGluon will flip their sign and print them as negative values during training (as it internally assumes higher values are better). You can even specify a custom metric by following the [Custom Metric Tutorial](advanced/tabular-custom-metric.ipynb).

**Data Formats:** AutoGluon can currently operate on data tables already loaded into Python as pandas DataFrames, or those stored in files of [CSV format](https://en.wikipedia.org/wiki/Comma-separated_values) or [Parquet format](https://databricks.com/glossary/what-is-parquet). If your data lives in multiple tables, you will first need to join them into a single table whose rows correspond to statistically independent observations (datapoints) and columns correspond to different features (aka. variables/covariates).

Refer to the [TabularPredictor documentation](../../api/autogluon.tabular.TabularPredictor.rst) to see all of the available methods/options.

## Advanced Usage

For more advanced usage examples of AutoGluon, refer to the [In Depth Tutorial](tabular-indepth.ipynb)

If you are interested in deployment optimization, refer to the [Deployment Optimization Tutorial](advanced/tabular-deployment.ipynb).

For adding custom models to AutoGluon, refer to the [Custom Model](advanced/tabular-custom-model.ipynb) and [Custom Model Advanced](advanced/tabular-custom-model-advanced.ipynb) tutorials.
