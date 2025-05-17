Summary: This tutorial demonstrates implementing multimodal machine learning using AutoGluon's MultiModalPredictor for combined text and tabular data analysis. It covers essential techniques for data preprocessing (including numeric conversions and log transformations), model configuration, training, prediction, and embedding extraction. The tutorial helps with tasks like automated feature fusion, mixed-type data handling, and neural network generation. Key functionalities include automatic architecture selection based on data types, joint training across modalities, embedding extraction for downstream tasks, and flexible model customization options. The implementation focuses on practical aspects like proper numerical preprocessing, time limit setting, and model artifact management.

# AutoMM for Text + Tabular - Quick Start

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/autogluon/autogluon/blob/master/docs/tutorials/multimodal/multimodal_prediction/multimodal_text_tabular.ipynb)
[![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/autogluon/autogluon/blob/master/docs/tutorials/multimodal/multimodal_prediction/multimodal_text_tabular.ipynb)



In many applications, text data may be mixed with numeric/categorical data. 
AutoGluon's `MultiModalPredictor` can train a single neural network that jointly operates on multiple feature types, 
including text, categorical, and numerical columns. The general idea is to embed the text, categorical and numeric fields 
separately and fuse these features across modalities. This tutorial demonstrates such an application.


```python
!pip install autogluon.multimodal

```


```python
import numpy as np
import pandas as pd
import warnings
import os

warnings.filterwarnings('ignore')
np.random.seed(123)
```


```python
!python3 -m pip install openpyxl
```

## Book Price Prediction Data

For demonstration, we use the book price prediction dataset from the [MachineHack Book Price Prediction Hackathon](https://machinehack.com/hackathons/predict_the_price_of_books/overview). Our goal is to predict a book's price given various features like its author, the abstract, the book's rating, etc.


```python
!mkdir -p price_of_books
!wget https://automl-mm-bench.s3.amazonaws.com/machine_hack_competitions/predict_the_price_of_books/Data.zip -O price_of_books/Data.zip
!cd price_of_books && unzip -o Data.zip
!ls price_of_books/Participants_Data
```


```python
train_df = pd.read_excel(os.path.join('price_of_books', 'Participants_Data', 'Data_Train.xlsx'), engine='openpyxl')
train_df.head()
```

We do some basic preprocessing to convert `Reviews` and `Ratings` in the data table to numeric values, and we transform prices to a log-scale.


```python
def preprocess(df):
    df = df.copy(deep=True)
    df.loc[:, 'Reviews'] = pd.to_numeric(df['Reviews'].apply(lambda ele: ele[:-len(' out of 5 stars')]))
    df.loc[:, 'Ratings'] = pd.to_numeric(df['Ratings'].apply(lambda ele: ele.replace(',', '')[:-len(' customer reviews')]))
    df.loc[:, 'Price'] = np.log(df['Price'] + 1)
    return df
```


```python
train_subsample_size = 1500  # subsample for faster demo, you can try setting to larger values
test_subsample_size = 5
train_df = preprocess(train_df)
train_data = train_df.iloc[100:].sample(train_subsample_size, random_state=123)
test_data = train_df.iloc[:100].sample(test_subsample_size, random_state=245)
train_data.head()
```

## Training

We can simply create a MultiModalPredictor and call `predictor.fit()` to train a model that operates on across all types of features. 
Internally, the neural network will be automatically generated based on the inferred data type of each feature column. 
To save time, we subsample the data and only train for three minutes.


```python
from autogluon.multimodal import MultiModalPredictor
import uuid

time_limit = 3 * 60  # set to larger value in your applications
model_path = f"./tmp/{uuid.uuid4().hex}-automm_text_book_price_prediction"
predictor = MultiModalPredictor(label='Price', path=model_path)
predictor.fit(train_data, time_limit=time_limit)
```

## Prediction

We can easily obtain predictions and extract data embeddings using the MultiModalPredictor.


```python
predictions = predictor.predict(test_data)
print('Predictions:')
print('------------')
print(np.exp(predictions) - 1)
print()
print('True Value:')
print('------------')
print(np.exp(test_data['Price']) - 1)

```


```python
performance = predictor.evaluate(test_data)
print(performance)
```


```python
embeddings = predictor.extract_embedding(test_data)
embeddings.shape
```


## Other Examples

You may go to [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm) to explore other examples about AutoMM.

## Customization
To learn how to customize AutoMM, please refer to [Customize AutoMM](../advanced_topics/customization.ipynb).
