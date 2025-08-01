Summary: This tutorial introduces Chronos and Chronos-Bolt models in AutoGluon for time series forecasting. It covers implementation of zero-shot forecasting that scales linearly with time series volume, model selection across different sizes (tiny to large), and practical code examples for prediction. Key functionalities include fine-tuning options with customizable learning rates and steps, incorporating covariates through regressors (particularly CatBoost), and hardware recommendations. The tutorial demonstrates how to load data, split into train/test sets, create predictors with various configurations, generate predictions, and evaluate model performance through leaderboards—particularly useful for implementing efficient time series forecasting with minimal training requirements.

```python
# We use uv for faster installation
!pip install uv
!uv pip install -q autogluon.timeseries --system
!uv pip uninstall -q torchaudio torchvision torchtext --system # fix incompatible package versions on Colab
```

## Getting started with Chronos

Being a pretrained model for zero-shot forecasting, Chronos is different from other models available in AG-TS. 
Specifically, Chronos models do not really `fit` time series data. However, when `predict` is called, they carry out a relatively more expensive computation that scales linearly with the number of time series in the dataset. In this aspect, they behave like local statistical models such as ETS or ARIMA, where all computation happens during inference. 

AutoGluon supports both the original Chronos models (e.g., [`chronos-t5-large`](https://huggingface.co/autogluon/chronos-t5-large)), as well as the new, more accurate and up to 250x faster Chronos-Bolt⚡ models (e.g., [`chronos-bolt-base`](https://huggingface.co/autogluon/chronos-bolt-base)). 

The easiest way to get started with Chronos is through the model-specific presets. 

- **(recommended)** The new, fast Chronos-Bolt️ models can be accessed using the `"bolt_tiny"`, `"bolt_mini"`, `"bolt_small"` and `"bolt_base"` presets.
- The original Chronos models can be accessed using the `"chronos_tiny"`, `"chronos_mini"`, `"chronos_small"`, `"chronos_base"` and `"chronos_large"` presets.

Note that the original Chronos models of size `small` and above require a GPU to run, while all Chronos-Bolt models can be run both on a CPU and a GPU.

Alternatively, Chronos can be combined with other time series models using presets `"medium_quality"`, `"high_quality"` and `"best_quality"`. More details about these presets are available in the documentation for [`TimeSeriesPredictor.fit`](https://auto.gluon.ai/stable/api/autogluon.timeseries.TimeSeriesPredictor.fit.html).

## Zero-shot forecasting

Let's work with a subset of the [Australian Electricity Demand dataset](https://zenodo.org/records/4659727) to see Chronos-Bolt in action.

First, we load the dataset as a [TimeSeriesDataFrame](https://auto.gluon.ai/stable/api/autogluon.timeseries.TimeSeriesDataFrame.html).


```python
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
```


```python
data = TimeSeriesDataFrame.from_path(
    "https://autogluon.s3.amazonaws.com/datasets/timeseries/australian_electricity_subset/test.csv"
)
data.head()
```

Next, we create the [TimeSeriesPredictor](https://auto.gluon.ai/stable/api/autogluon.timeseries.TimeSeriesPredictor.html) and select the `"bolt_small"` presets to use the Chronos-Bolt (Small, 48M) model in zero-shot mode.


```python
prediction_length = 48
train_data, test_data = data.train_test_split(prediction_length)

predictor = TimeSeriesPredictor(prediction_length=prediction_length).fit(
    train_data, presets="bolt_small",
)
```

As promised, Chronos does not take any time to `fit`. The `fit` call merely serves as a proxy for the `TimeSeriesPredictor` to do some of its chores under the hood, such as inferring the frequency of time series and saving the predictor's state to disk. 

Let's use the `predict` method to generate forecasts, and the `plot` method to visualize them.


```python
predictions = predictor.predict(train_data)
predictor.plot(
    data=data,
    predictions=predictions,
    item_ids=data.item_ids[:2],
    max_history_length=200,
);
```

## Fine-tuning 

We have seen above how Chronos models can produce forecasts in zero-shot mode. AutoGluon also makes it easy to fine-tune Chronos models on a specific dataset to maximize the predictive accuracy.

The following snippet specifies two settings for the Chronos-Bolt ️(Small) model: zero-shot and fine-tuned. `TimeSeriesPredictor` will perform a lightweight fine-tuning of the pretrained model on the provided training data. We add name suffixes to easily identify the zero-shot and fine-tuned versions of the model.


```python
predictor = TimeSeriesPredictor(prediction_length=prediction_length).fit(
    train_data=train_data,
    hyperparameters={
        "Chronos": [
            {"model_path": "bolt_small", "ag_args": {"name_suffix": "ZeroShot"}},
            {"model_path": "bolt_small", "fine_tune": True, "ag_args": {"name_suffix": "FineTuned"}},
        ]
    },
    time_limit=60,  # time limit in seconds
    enable_ensemble=False,
)
```

Here we used the default fine-tuning configuration for Chronos by only specifying `"fine_tune": True`. However, AutoGluon makes it easy to change other parameters for fine-tuning such as the number of steps or learning rate.
```python
predictor.fit(
    ...,
    hyperparameters={"Chronos": {"fine_tune": True, "fine_tune_lr": 1e-4, "fine_tune_steps": 2000}},
)
```

For the full list of fine-tuning options, see the Chronos documentation in [Forecasting Model Zoo](forecasting-model-zoo.md#autogluon.timeseries.models.ChronosModel).


After fitting, we can evaluate the two model variants on the test data and generate a leaderboard.


```python
predictor.leaderboard(test_data)
```

Fine-tuning resulted in a more accurate model, as shown by the better `score_test` on the test set.

Note that all AutoGluon-TimeSeries models report scores in a "higher is better" format, meaning that most forecasting error metrics like WQL are multiplied by -1 when reported.

## Incorporating the covariates

Chronos️ is a univariate model, meaning it relies solely on the historical data of the target time series for making predictions. However, in real-world scenarios, additional exogenous information related to the target series (e.g., holidays, promotions) is often available. Leveraging this information when making predictions can improve forecast accuracy. 

AG-TS now features covariate regressors that can be combined with univariate models like Chronos-Bolt to incorporate exogenous information. 
A `covariate_regressor` in AG-TS is a tabular regression model that is fit on the known covariates and static features to predict the target column at the each time step. The predictions of the covariate regressor are subtracted from the target column, and the univariate model then forecasts the residuals.


```python
data = TimeSeriesDataFrame.from_path(
    "https://autogluon.s3.amazonaws.com/datasets/timeseries/grocery_sales/test.csv",
)
data.head()
```

We use a grocery sales dataset to demonstrate how Chronos-Bolt can be combined with a covariate regressor. This dataset includes 3 known covariates: `scaled_price`, `promotion_email` and `promotion_homepage` and the task is to forecast the `unit_sales`.


```python
prediction_length = 8
train_data, test_data = data.train_test_split(prediction_length=prediction_length)
```

The following code fits a TimeSeriesPredictor to forecast `unit_sales` for the next 8 weeks. 

Note that we have specified the target column we are interested in forecasting and the names of known covariates while constructing the TimeSeriesPredictor. 

We define two configurations for Chronos-Bolt: 
- zero-shot configuration that uses only the historical values of `unit_sales` without considering the covariates;
- a configuration with a CatBoost regression model as the `covariate_regressor`. Note that we recommend to apply a `target_scaler` when using a covariate regressor. Target scaler ensures that all time series have comparable scales, often leading to better accuracy.

Like before, we add suffixes to model names to more easily distinguish them in the leaderboard.


```python
predictor = TimeSeriesPredictor(
    prediction_length=prediction_length,
    target="unit_sales",
    known_covariates_names=["scaled_price", "promotion_email", "promotion_homepage"],
).fit(
    train_data,
    hyperparameters={
        "Chronos": [
            # Zero-shot model WITHOUT covariates
            {
                "model_path": "bolt_small",
                "ag_args": {"name_suffix": "ZeroShot"},
            },
            # Chronos-Bolt (Small) combined with CatBoost on covariates
            {
                "model_path": "bolt_small",
                "covariate_regressor": "CAT",
                "target_scaler": "standard",
                "ag_args": {"name_suffix": "WithRegressor"},
            },
        ],
    },
    enable_ensemble=False,
    time_limit=60,
)
```

Once the predictor has been fit, we can evaluate it on the test dataset and generate the leaderboard. We see that the model that utilizes the covariates produces a more accurate forecast on the test set.


```python
predictor.leaderboard(test_data)
```

Note that the covariates may not always be useful — for some datasets, the zero-shot model may achieve better accuracy. Therefore, it's always important to try out multiple models and select the one that achieves the best accuracy on held-out data. This is done automatically in AutoGluon's `"high_quality"` and `"best_quality"` presets.

## FAQ


#### How accurate is Chronos?

In several independent evaluations we found Chronos to be effective in zero-shot forecasting. 
The accuracy of Chronos-Bolt (base) often exceeds statistical baseline models, and is often comparable to deep learning 
models such as `TemporalFusionTransformer` or `PatchTST`.

#### What is the recommended hardware for running Chronos models?

For fine-tuning and inference with larger Chronos and Chronos-Bolt models, we tested the AWS `g5.2xlarge` and `p3.2xlarge` instances that feature NVIDIA A10G and V100 GPUs, with at least 16GiB of GPU memory and 32GiB of main memory. 

Chronos-Bolt models can also be used on CPU machines, but this will typically result in a longer runtime.


#### Where can I ask specific questions on Chronos?

The AutoGluon team are among the core developers of Chronos. So you can ask Chronos-related questions on AutoGluon channels such 
as the Discord [server](https://discord.gg/wjUmjqAc2N), or [GitHub](https://github.com/autogluon/autogluon). You can also join 
the discussion on the Chronos GitHub [page](https://github.com/amazon-science/chronos-forecasting/discussions).
