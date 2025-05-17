Summary: This tutorial demonstrates AutoGluon's MultiModalPredictor implementation for image classification tasks, covering essential techniques for model training, evaluation, and deployment. It provides code examples for data preparation (supporting both image paths and bytearrays), model training configuration, prediction generation, feature extraction, and model persistence. Key functionalities include accuracy evaluation, probability predictions, embedding extraction (512-2048 dimensional vectors), and flexible input handling. The tutorial is particularly useful for implementing automated image classification pipelines, with specific focus on practical aspects like model saving/loading, security considerations, and input format compatibility.

# AutoMM for Image Classification - Quick Start

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/autogluon/autogluon/blob/master/docs/tutorials/multimodal/image_prediction/beginner_image_cls.ipynb)
[![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/autogluon/autogluon/blob/master/docs/tutorials/multimodal/image_prediction/beginner_image_cls.ipynb)



In this quick start, we'll use the task of image classification to illustrate how to use **MultiModalPredictor**. Once the data is prepared in [Pandas DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html) format, a single call to `MultiModalPredictor.fit()` will take care of the model training for you.


## Create Image Dataset

For demonstration purposes, we use a subset of the [Shopee-IET dataset](https://www.kaggle.com/competitions/demo-shopee-iet-competition/data) from Kaggle.
Each image in this data depicts a clothing item and the corresponding label specifies its clothing category.
Our subset of the data contains the following possible labels: `BabyPants`, `BabyShirt`, `womencasualshoes`, `womenchiffontop`.

We can load a dataset by downloading a url data automatically:


```python
!pip install autogluon.multimodal

```


```python
import warnings
warnings.filterwarnings('ignore')
import pandas as pd

from autogluon.multimodal.utils.misc import shopee_dataset
download_dir = './ag_automm_tutorial_imgcls'
train_data_path, test_data_path = shopee_dataset(download_dir)
print(train_data_path)
```

We can see there are 800 rows and 2 columns in this training dataframe. The 2 columns are **image** and **label**, and the **image** column contains the absolute paths of the images. Each row represents a different training sample.

In addition to image paths, `MultiModalPredictor` also supports image bytearrays during training and inference. We can load the dataset with bytearrays with the option `is_bytearray` set to `True`:


```python
import warnings
warnings.filterwarnings('ignore')

download_dir = './ag_automm_tutorial_imgcls'
train_data_byte, test_data_byte = shopee_dataset(download_dir, is_bytearray=True)
```

## Use AutoMM to Fit Models

Now, we fit a classifier using AutoMM as follows:


```python
from autogluon.multimodal import MultiModalPredictor
import uuid
model_path = f"./tmp/{uuid.uuid4().hex}-automm_shopee"
predictor = MultiModalPredictor(label="label", path=model_path)
predictor.fit(
    train_data=train_data_path,
    time_limit=30, # seconds
)
```

**label** is the name of the column that contains the target variable to predict, e.g., it is "label" in our example. **path** indicates the directory where models and intermediate outputs should be saved. We set the training time limit to 30 seconds for demonstration purpose, but you can control the training time by setting configurations. To customize AutoMM, please refer to [Customize AutoMM](../advanced_topics/customization.ipynb).


## Evaluate on Test Dataset

You can evaluate the classifier on the test dataset to see how it performs, the test top-1 accuracy is:


```python
scores = predictor.evaluate(test_data_path, metrics=["accuracy"])
print('Top-1 test acc: %.3f' % scores["accuracy"])
```

You can also evaluate on test data with image bytearray using the model trained on training data with image path, and vice versa:


```python
scores = predictor.evaluate(test_data_byte, metrics=["accuracy"])
print('Top-1 test acc: %.3f' % scores["accuracy"])
```

## Predict on a New Image

Given an example image, let's visualize it first,


```python
image_path = test_data_path.iloc[0]['image']
from IPython.display import Image, display
pil_img = Image(filename=image_path)
display(pil_img)
```

We can easily use the final model to `predict` the label,


```python
predictions = predictor.predict({'image': [image_path]})
print(predictions)
```

If probabilities of all categories are needed, you can call `predict_proba`:


```python
proba = predictor.predict_proba({'image': [image_path]})
print(proba)
```

Similarly as `predictor.evaluate`, we can also parse image_bytearrays into `.predict` and `.predict_proba`:


```python
image_byte = test_data_byte.iloc[0]['image']
predictions = predictor.predict({'image': [image_byte]})
print(predictions)

proba = predictor.predict_proba({'image': [image_byte]})
print(proba)
```

## Extract Embeddings

Extracting representation from the whole image learned by a model is also very useful. We provide `extract_embedding` function to allow predictor to return the N-dimensional image feature where `N` depends on the model(usually a 512 to 2048 length vector)


```python
feature = predictor.extract_embedding({'image': [image_path]})
print(feature[0].shape)
```

You should expect the same result when extract embedding from image bytearray:


```python
feature = predictor.extract_embedding({'image': [image_byte]})
print(feature[0].shape)
```

## Save and Load

The trained predictor is automatically saved at the end of `fit()`, and you can easily reload it.

```{warning}

`MultiModalPredictor.load()` uses `pickle` module implicitly, which is known to be insecure. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling. Never load data that could have come from an untrusted source, or that could have been tampered with. **Only load data you trust.**

```


```python
loaded_predictor = MultiModalPredictor.load(model_path)
load_proba = loaded_predictor.predict_proba({'image': [image_path]})
print(load_proba)
```

We can see the predicted class probabilities are still the same as above, which means same model!

## Other Examples

You may go to [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm) to explore other examples about AutoMM.

## Customization
To learn how to customize AutoMM, please refer to [Customize AutoMM](../advanced_topics/customization.ipynb).
