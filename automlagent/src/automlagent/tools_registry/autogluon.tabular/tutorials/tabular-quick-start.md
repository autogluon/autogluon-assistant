Summary: This tutorial demonstrates AutoGluon's tabular machine learning implementation, focusing on automated model training and prediction workflows. It covers essential techniques for loading tabular data, training models with customizable time limits, and evaluating model performance using TabularPredictor. The tutorial helps with tasks like automated feature engineering, model selection, and ensemble creation for both classification and regression problems. Key features include built-in data type handling, automatic model selection, hyperparameter tuning, and performance evaluation through leaderboards, all achievable with minimal code requirements. The implementation emphasizes AutoGluon's ability to handle complex ML pipelines with simple API calls while supporting advanced customization options for features, models, and metrics.

# AutoGluon Tabular - Quick Start

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/autogluon/autogluon/blob/master/docs/tutorials/tabular/tabular-quick-start.ipynb)
[![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/autogluon/autogluon/blob/master/docs/tutorials/tabular/tabular-quick-start.ipynb)

In this tutorial, we will see how to use AutoGluon's `TabularPredictor` to predict the values of a target column based on the other columns in a tabular dataset.

Begin by making sure AutoGluon is installed, and then import AutoGluon's `TabularDataset` and `TabularPredictor`. We will use the former to load data and the latter to train models and make predictions. 


```python
!python -m pip install --upgrade pip
!python -m pip install autogluon
```


```python
from autogluon.tabular import TabularDataset, TabularPredictor
```

## Example Data

For this tutorial we will use a dataset from the cover story of [Nature issue 7887](https://www.nature.com/nature/volumes/600/issues/7887): [AI-guided intuition for math theorems](https://www.nature.com/articles/s41586-021-04086-x.pdf). The goal is to predict a knot's signature based on its properties. We sampled 10K training and 5K test examples from the [original data](https://github.com/deepmind/mathematics_conjectures/blob/main/knot_theory.ipynb). The sampled dataset make this tutorial run quickly, but AutoGluon can handle the full dataset if desired.

We load this dataset directly from a URL. AutoGluon's `TabularDataset` is a subclass of pandas [DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html), so any `DataFrame` methods can be used on `TabularDataset` as well.


```python
data_url = 'https://raw.githubusercontent.com/mli/ag-docs/main/knot_theory/'
train_data = TabularDataset(f'{data_url}train.csv')
train_data.head()
```

Our targets are stored in the "signature" column, which has 18 unique integers. Even though pandas didn't correctly recognize this data type as categorical, AutoGluon will fix this issue.



```python
label = 'signature'
train_data[label].describe()
```

## Training

We now construct a `TabularPredictor` by specifying the label column name and then train on the dataset with `TabularPredictor.fit()`. We don't need to specify any other parameters. AutoGluon will recognize this is a multi-class classification task, perform automatic feature engineering, train multiple models, and then ensemble the models to create the final predictor. 


```python
predictor = TabularPredictor(label=label).fit(train_data)
```

Model fitting should take a few minutes or less depending on your CPU. You can make training faster by specifying the `time_limit` argument. For example, `fit(..., time_limit=60)` will stop training after 60 seconds. Higher time limits will generally result in better prediction performance, and excessively low time limits will prevent AutoGluon from training and ensembling a reasonable set of models.



## Prediction

Once we have a predictor that is fit on the training dataset, we can load a separate set of data to use for prediction and evaulation.


```python
test_data = TabularDataset(f'{data_url}test.csv')

y_pred = predictor.predict(test_data.drop(columns=[label]))
y_pred.head()
```

## Evaluation

We can evaluate the predictor on the test dataset using the `evaluate()` function, which measures how well our predictor performs on data that was not used for fitting the models.


```python
predictor.evaluate(test_data, silent=True)
```

AutoGluon's `TabularPredictor` also provides the `leaderboard()` function, which allows us to evaluate the performance of each individual trained model on the test data.


```python
predictor.leaderboard(test_data)
```

## Conclusion

In this quickstart tutorial we saw AutoGluon's basic fit and predict functionality using `TabularDataset` and `TabularPredictor`. AutoGluon simplifies the model training process by not requiring feature engineering or model hyperparameter tuning. Check out the in-depth tutorials to learn more about AutoGluon's other features like customizing the training and prediction steps or extending AutoGluon with custom feature generators, models, or metrics.
