Summary: This tutorial covers implementing time series forecasting using AutoGluon's Chronos models, specifically focusing on the faster Chronos-Bolt variants. It demonstrates how to perform zero-shot forecasting and model fine-tuning, incorporate covariates using regression models, and compare model performances. Key implementation techniques include basic model setup, covariate integration through tabular regressors, and fine-tuning configurations with customizable learning rates and steps. The tutorial helps with tasks like time series prediction, model optimization, and handling exogenous variables. Notable features include support for both CPU and GPU execution, various model sizes (tiny to large), automated model selection through presets, and visualization capabilities. It emphasizes best practices for model selection, hardware requirements, and performance optimization.

# Forecasting with Chronos

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/autogluon/autogluon/blob/master/docs/tutorials/timeseries/forecasting-chronos.ipynb)
[![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/autogluon/autogluon/blob/master/docs/tutorials/timeseries/forecasting-chronos.ipynb)


AutoGluon-TimeSeries (AG-TS) includes [Chronos](https://github.com/amazon-science/chronos-forecasting) family of forecasting models. Chronos models are pretrained on a large collection of real & synthetic time series data, which enables them to make accurate forecasts on new data out of the box.

AG-TS provides a robust and easy way to use Chronos through the familiar `TimeSeriesPredictor` API. This tutorial describes how to 
- Use Chronos models in **zero-shot** mode to make forecasts without any dataset-specific training
- **Fine-tune** Chronos models on custom data to improve the accuracy
- Handle **covariates & static features** by combining Chronos with a tabular regression model

:::{note}

**New in v1.2:** AutoGluon now features Chronos-Bolt⚡️ — new, more accurate, and up to 250x faster Chronos models.

:::


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




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>target</th>
    </tr>
    <tr>
      <th>item_id</th>
      <th>timestamp</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">T000000</th>
      <th>2013-03-10 00:00:00</th>
      <td>5207.959961</td>
    </tr>
    <tr>
      <th>2013-03-10 00:30:00</th>
      <td>5002.275879</td>
    </tr>
    <tr>
      <th>2013-03-10 01:00:00</th>
      <td>4747.569824</td>
    </tr>
    <tr>
      <th>2013-03-10 01:30:00</th>
      <td>4544.880859</td>
    </tr>
    <tr>
      <th>2013-03-10 02:00:00</th>
      <td>4425.952148</td>
    </tr>
  </tbody>
</table>
</div>



Next, we create the [TimeSeriesPredictor](https://auto.gluon.ai/stable/api/autogluon.timeseries.TimeSeriesPredictor.html) and select the `"bolt_small"` presets to use the Chronos-Bolt (Small, 48M) model in zero-shot mode.


```python
prediction_length = 48
train_data, test_data = data.train_test_split(prediction_length)

predictor = TimeSeriesPredictor(prediction_length=prediction_length).fit(
    train_data, presets="bolt_small",
)
```

    Sorting the dataframe index before generating the train/test split.
    Beginning AutoGluon training...
    AutoGluon will save models to '/local/home/shchuro/workspace/autogluon/docs/tutorials/timeseries/AutogluonModels/ag-20241126_091557'
    =================== System Info ===================
    AutoGluon Version:  1.1.2b20241122
    Python Version:     3.11.10
    Operating System:   Linux
    Platform Machine:   x86_64
    Platform Version:   #1 SMP Wed Oct 23 01:22:11 UTC 2024
    CPU Count:          32
    GPU Count:          4
    Memory Avail:       230.68 GB / 239.85 GB (96.2%)
    Disk Space Avail:   563.58 GB / 1968.52 GB (28.6%)
    ===================================================
    Setting presets to: bolt_small
    
    Fitting with arguments:
    {'enable_ensemble': True,
     'eval_metric': WQL,
     'hyperparameters': {'Chronos': {'model_path': 'bolt_small'}},
     'known_covariates_names': [],
     'num_val_windows': 1,
     'prediction_length': 48,
     'quantile_levels': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
     'random_seed': 123,
     'refit_every_n_windows': 1,
     'refit_full': False,
     'skip_model_selection': True,
     'target': 'target',
     'verbosity': 2}
    
    Inferred time series frequency: '30min'
    Provided train_data has 172800 rows, 5 time series. Median time series length is 34560 (min=34560, max=34560). 
    
    Provided data contains following columns:
    	target: 'target'
    
    AutoGluon will gauge predictive performance using evaluation metric: 'WQL'
    	This metric's sign has been flipped to adhere to being higher_is_better. The metric score can be multiplied by -1 to get the metric value.
    ===================================================
    
    Starting training. Start time is 2024-11-26 09:16:01
    Models that will be trained: ['Chronos[bolt_small]']
    Training timeseries model Chronos[bolt_small]. 
    	2.01    s     = Training runtime
    Training complete. Models trained: ['Chronos[bolt_small]']
    Total runtime: 2.01 s
    Best model: Chronos[bolt_small]


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

    Model not specified in predict, will default to the model with the best validation score: Chronos[bolt_small]



    
![png](output_10_1.png)
    


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

    Beginning AutoGluon training... Time limit = 60s
    AutoGluon will save models to '/local/home/shchuro/workspace/autogluon/docs/tutorials/timeseries/AutogluonModels/ag-20241126_091607'
    =================== System Info ===================
    AutoGluon Version:  1.1.2b20241122
    Python Version:     3.11.10
    Operating System:   Linux
    Platform Machine:   x86_64
    Platform Version:   #1 SMP Wed Oct 23 01:22:11 UTC 2024
    CPU Count:          32
    GPU Count:          4
    Memory Avail:       229.82 GB / 239.85 GB (95.8%)
    Disk Space Avail:   563.58 GB / 1968.52 GB (28.6%)
    ===================================================
    
    Fitting with arguments:
    {'enable_ensemble': False,
     'eval_metric': WQL,
     'hyperparameters': {'Chronos': [{'ag_args': {'name_suffix': 'ZeroShot'},
                                      'model_path': 'bolt_small'},
                                     {'ag_args': {'name_suffix': 'FineTuned'},
                                      'fine_tune': True,
                                      'model_path': 'bolt_small'}]},
     'known_covariates_names': [],
     'num_val_windows': 1,
     'prediction_length': 48,
     'quantile_levels': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
     'random_seed': 123,
     'refit_every_n_windows': 1,
     'refit_full': False,
     'skip_model_selection': False,
     'target': 'target',
     'time_limit': 60,
     'verbosity': 2}
    
    Inferred time series frequency: '30min'
    Provided train_data has 172800 rows, 5 time series. Median time series length is 34560 (min=34560, max=34560). 
    
    Provided data contains following columns:
    	target: 'target'
    
    AutoGluon will gauge predictive performance using evaluation metric: 'WQL'
    	This metric's sign has been flipped to adhere to being higher_is_better. The metric score can be multiplied by -1 to get the metric value.
    ===================================================
    
    Starting training. Start time is 2024-11-26 09:16:07
    Models that will be trained: ['ChronosZeroShot[bolt_small]', 'ChronosFineTuned[bolt_small]']
    Training timeseries model ChronosZeroShot[bolt_small]. Training for up to 29.9s of the 59.9s of remaining time.
    	-0.0417       = Validation score (-WQL)
    	0.10    s     = Training runtime
    	0.83    s     = Validation (prediction) runtime
    Training timeseries model ChronosFineTuned[bolt_small]. Training for up to 59.0s of the 59.0s of remaining time.
    	Saving fine-tuned model to /local/home/shchuro/workspace/autogluon/docs/tutorials/timeseries/AutogluonModels/ag-20241126_091607/models/ChronosFineTuned[bolt_small]/W0/fine-tuned-ckpt
    	-0.0290       = Validation score (-WQL)
    	49.36   s     = Training runtime
    	0.07    s     = Validation (prediction) runtime
    Training complete. Models trained: ['ChronosZeroShot[bolt_small]', 'ChronosFineTuned[bolt_small]']
    Total runtime: 50.38 s
    Best model: ChronosFineTuned[bolt_small]
    Best model score: -0.0290


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

    Additional data provided, testing on additional data. Resulting leaderboard will be sorted according to test score (`score_test`).





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>score_test</th>
      <th>score_val</th>
      <th>pred_time_test</th>
      <th>pred_time_val</th>
      <th>fit_time_marginal</th>
      <th>fit_order</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ChronosFineTuned[bolt_small]</td>
      <td>-0.030785</td>
      <td>-0.029021</td>
      <td>0.541208</td>
      <td>0.073925</td>
      <td>49.362413</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ChronosZeroShot[bolt_small]</td>
      <td>-0.041446</td>
      <td>-0.041720</td>
      <td>0.859698</td>
      <td>0.825092</td>
      <td>0.098496</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



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




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>scaled_price</th>
      <th>promotion_email</th>
      <th>promotion_homepage</th>
      <th>unit_sales</th>
    </tr>
    <tr>
      <th>item_id</th>
      <th>timestamp</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">1062_101</th>
      <th>2018-01-01</th>
      <td>0.879130</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>636.0</td>
    </tr>
    <tr>
      <th>2018-01-08</th>
      <td>0.994517</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>123.0</td>
    </tr>
    <tr>
      <th>2018-01-15</th>
      <td>1.005513</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>391.0</td>
    </tr>
    <tr>
      <th>2018-01-22</th>
      <td>1.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>339.0</td>
    </tr>
    <tr>
      <th>2018-01-29</th>
      <td>0.883309</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>661.0</td>
    </tr>
  </tbody>
</table>
</div>



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

    Beginning AutoGluon training... Time limit = 60s
    AutoGluon will save models to '/local/home/shchuro/workspace/autogluon/docs/tutorials/timeseries/AutogluonModels/ag-20241126_091700'
    =================== System Info ===================
    AutoGluon Version:  1.1.2b20241122
    Python Version:     3.11.10
    Operating System:   Linux
    Platform Machine:   x86_64
    Platform Version:   #1 SMP Wed Oct 23 01:22:11 UTC 2024
    CPU Count:          32
    GPU Count:          4
    Memory Avail:       229.61 GB / 239.85 GB (95.7%)
    Disk Space Avail:   563.40 GB / 1968.52 GB (28.6%)
    ===================================================
    
    Fitting with arguments:
    {'enable_ensemble': False,
     'eval_metric': WQL,
     'hyperparameters': {'Chronos': [{'ag_args': {'name_suffix': 'ZeroShot'},
                                      'model_path': 'bolt_small'},
                                     {'ag_args': {'name_suffix': 'WithRegressor'},
                                      'covariate_regressor': 'CAT',
                                      'model_path': 'bolt_small',
                                      'target_scaler': 'standard'}]},
     'known_covariates_names': ['scaled_price',
                                'promotion_email',
                                'promotion_homepage'],
     'num_val_windows': 1,
     'prediction_length': 8,
     'quantile_levels': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
     'random_seed': 123,
     'refit_every_n_windows': 1,
     'refit_full': False,
     'skip_model_selection': False,
     'target': 'unit_sales',
     'time_limit': 60,
     'verbosity': 2}
    
    Inferred time series frequency: 'W-MON'
    Provided train_data has 7337 rows (NaN fraction=6.6%), 319 time series. Median time series length is 23 (min=23, max=23). 
    
    Provided data contains following columns:
    	target: 'unit_sales'
    	known_covariates:
    		categorical:        []
    		continuous (float): ['scaled_price', 'promotion_email', 'promotion_homepage']
    
    To learn how to fix incorrectly inferred types, please see documentation for TimeSeriesPredictor.fit
    
    AutoGluon will gauge predictive performance using evaluation metric: 'WQL'
    	This metric's sign has been flipped to adhere to being higher_is_better. The metric score can be multiplied by -1 to get the metric value.
    ===================================================
    
    Starting training. Start time is 2024-11-26 09:17:00
    Models that will be trained: ['ChronosZeroShot[bolt_small]', 'ChronosWithRegressor[bolt_small]']
    Training timeseries model ChronosZeroShot[bolt_small]. Training for up to 29.9s of the 59.9s of remaining time.
    	-0.4523       = Validation score (-WQL)
    	0.02    s     = Training runtime
    	0.84    s     = Validation (prediction) runtime
    Training timeseries model ChronosWithRegressor[bolt_small]. Training for up to 59.0s of the 59.0s of remaining time.
    	-0.3580       = Validation score (-WQL)
    	1.00    s     = Training runtime
    	0.92    s     = Validation (prediction) runtime
    Training complete. Models trained: ['ChronosZeroShot[bolt_small]', 'ChronosWithRegressor[bolt_small]']
    Total runtime: 2.80 s
    Best model: ChronosWithRegressor[bolt_small]
    Best model score: -0.3580


Once the predictor has been fit, we can evaluate it on the test dataset and generate the leaderboard. We see that the model that utilizes the covariates produces a more accurate forecast on the test set.


```python
predictor.leaderboard(test_data)
```

    Additional data provided, testing on additional data. Resulting leaderboard will be sorted according to test score (`score_test`).





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>score_test</th>
      <th>score_val</th>
      <th>pred_time_test</th>
      <th>pred_time_val</th>
      <th>fit_time_marginal</th>
      <th>fit_order</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ChronosWithRegressor[bolt_small]</td>
      <td>-0.268969</td>
      <td>-0.358048</td>
      <td>0.881176</td>
      <td>0.916053</td>
      <td>1.004376</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ChronosZeroShot[bolt_small]</td>
      <td>-0.318562</td>
      <td>-0.452296</td>
      <td>0.859930</td>
      <td>0.844927</td>
      <td>0.019435</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



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
