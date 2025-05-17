Summary: This tutorial demonstrates the implementation of AutoGluon's MultiModalPredictor for multimodal machine learning tasks, specifically focusing on combining image, text, and tabular data for classification problems. It provides code for data preparation (handling image paths and DataFrame formatting), model initialization, training, and prediction workflows. Key functionalities covered include automatic problem type detection, feature modality detection, model selection, and late-fusion model addition. The tutorial is particularly useful for tasks requiring unified processing of multiple data types, offering implementation details for prediction, probability scoring, and performance evaluation, while also highlighting advanced features like embedding extraction and model distillation.

# AutoGluon Multimodal - Quick Start

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/autogluon/autogluon/blob/master/docs/tutorials/multimodal/multimodal_prediction/multimodal-quick-start.ipynb)
[![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/autogluon/autogluon/blob/master/docs/tutorials/multimodal/multimodal_prediction/multimodal-quick-start.ipynb)

AutoGluon's `MultiModalPredictor` is a deep learning model zoo of model zoos that can automatically build state-of-the-art deep learning models for inputs including images, text, and tabular data. Convert your data into AutoGluon's multimodal dataframe format, and `MultiModalPredictor` can predict the values of one column based on the other features.

Begin by making sure AutoGluon is installed, and then import the required modules.


```python
!python -m pip install --upgrade pip
!python -m pip install autogluon
```


```python
import os
import warnings

import numpy as np

warnings.filterwarnings('ignore')
np.random.seed(123)
```

## Example Data

For this tutorial we use a simplified and subsampled version of the [PetFinder dataset](https://www.kaggle.com/c/petfinder-adoption-prediction). The goal is to predict pet adoption rates based on their adoption profiles. In this simplified version, the adoption speed is grouped into two categories: 0 (slow) and 1 (fast). We begin by downloading a zip file containing the petfinder datasets and unzipping them in the current working directory.


```python
from autogluon.core.utils.loaders import load_zip

download_dir = './ag_multimodal_tutorial'
zip_file = 'https://automl-mm-bench.s3.amazonaws.com/petfinder_for_tutorial.zip'

load_zip.unzip(zip_file, unzip_dir=download_dir)
```

Next, we use pandas to read the dataset's CSV files into `DataFrames`, noting that the column we are interested in learning to predict is "AdoptionSpeed".


```python
import pandas as pd

dataset_path = f'{download_dir}/petfinder_for_tutorial'

train_data = pd.read_csv(f'{dataset_path}/train.csv', index_col=0)
test_data = pd.read_csv(f'{dataset_path}/test.csv', index_col=0)

label_col = 'AdoptionSpeed'
```

The PetFinder dataset comes with a directory of images, and some records in the data have multiple images associated with them. AutoGluon's multimodal dataframe format requires that image columns contain a string whose value is a path to a single image file. For this example, we will limit the image feature column to only the first image and will need to do some path manipulations to get everything setup correctly for the current directory structure.


```python
image_col = 'Images'

train_data[image_col] = train_data[image_col].apply(lambda ele: ele.split(';')[0])
test_data[image_col] = test_data[image_col].apply(lambda ele: ele.split(';')[0])

def path_expander(path, base_folder):
    path_l = path.split(';')
    return ';'.join([os.path.abspath(os.path.join(base_folder, path)) for path in path_l])

train_data[image_col] = train_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
test_data[image_col] = test_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
```

Each animal's adoption profile includes pictures, a text description, and various tabular features such as age, breed, name, color, and more. Let's look at a picture and description for an example row of data.


```python
example_row = train_data.iloc[0]
example_image = example_row[image_col]

from IPython.display import Image, display
pil_img = Image(filename=example_image)
display(pil_img)

example_row['Description']
```

## Training

Now that the data is in a suitable format, we can fit `MultiModalPredictor` on the training data. Here we set a tight training time budget for this quick demo. More training time will lead to better prediction performance, but we can get surprisingly good performance in a short amount of time.


```python
from autogluon.multimodal import MultiModalPredictor

predictor = MultiModalPredictor(label=label_col).fit(
    train_data=train_data,
    time_limit=120
)
```

Under the hood `MultiModalPredictor` automatically infers the problem type (classification or regression), detects feature modalities, selects models from the multimodal model pools, and trains the selected models. If multiple backbones are used, MultiModalPredictor appends a late-fusion model (MLP or transformer) on top of them.

## Prediction

After fitting the model, we want to use it to predict the labels in the witheld test dataset.


```python
predictions = predictor.predict(test_data.drop(columns=label_col))
predictions[:5]
```

For classification tasks, we can just as easily get the prediction probabilities for each output class.


```python
probs = predictor.predict_proba(test_data.drop(columns=label_col))
probs[:5]
```

## Evaluation

Finally, we can evaluate the predictor on the witheld test dataset on other performance metrics, in this case [roc_auc](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html).


```python
scores = predictor.evaluate(test_data, metrics=["roc_auc"])
scores
```

## Conclusion

In this quickstart tutorial we saw the basic fit and predict functionality of AutoGluon's `MultiModalPredictor`, but we just scratched the surface on its functionality. Check out the in-depth tutorials to learn about other features of AutoGluon's `MultiModalPredictor` like embedding extraction, distillation, model fine-tuning, text or image prediction, and semantic matching.


