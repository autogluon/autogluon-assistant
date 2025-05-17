Summary: This tutorial demonstrates implementing few-shot learning using AutoGluon's MultiModalPredictor for both text and image classification tasks. It provides code examples for setting up few-shot classifiers with essential configurations like problem_type="few_shot_classification", label specification, and evaluation metrics. The implementation combines foundation model features with SVM classifiers, making it particularly effective for small datasets with limited samples per class. Key functionalities include comparing few-shot vs. standard classification approaches, handling both text and image modalities, and configuring the classifier with best practices for optimal performance on small datasets.

# Few Shot Learning with AutoMM

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/autogluon/autogluon/blob/master/docs/tutorials/multimodal/advanced_topics/few_shot_learning.ipynb)
[![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/autogluon/autogluon/blob/master/docs/tutorials/multimodal/advanced_topics/few_shot_learning.ipynb)


In this tutorial we introduce a simple but effective way for few shot classification problems. 
We present the functionality which leverages the high-quality features from foundation models and uses SVM for few shot classification tasks.
Specifically, we extract sample features with pretrained models, and use the features for SVM learning.
We show the effectiveness of the foundation-model-followed-by-SVM on a text classification dataset and an image classification dataset.

## Few Shot Text Classification
### Prepare Text Data
We prepare all datasets in the format of `pd.DataFrame` as in many of our tutorials have done.
For this tutorial, we'll use a small `MLDoc` dataset for demonstration.
The dataset is a text classification dataset, which contains 4 classes and we downsampled the training data to 10 samples per class, a.k.a 10 shots.
For more details regarding `MLDoc` please see this [link](https://github.com/facebookresearch/MLDoc).


```python
import pandas as pd
import os
from autogluon.core.utils.loaders import load_zip

download_dir = "./ag_automm_tutorial_fs_cls"
zip_file = "https://automl-mm-bench.s3.amazonaws.com/nlp_datasets/MLDoc-10shot-en.zip"
load_zip.unzip(zip_file, unzip_dir=download_dir)
dataset_path = os.path.join(download_dir)
train_df = pd.read_csv(f"{dataset_path}/train.csv", names=["label", "text"])
test_df = pd.read_csv(f"{dataset_path}/test.csv", names=["label", "text"])
print(train_df)
print(test_df)
```

### Train a Few Shot Classifier
In order to perform few shot classification, we need to use the `few_shot_classification` problem type.


```python
from autogluon.multimodal import MultiModalPredictor

predictor_fs_text = MultiModalPredictor(
    problem_type="few_shot_classification",
    label="label",  # column name of the label
    eval_metric="acc",
)
predictor_fs_text.fit(train_df)
scores = predictor_fs_text.evaluate(test_df, metrics=["acc", "f1_macro"])
print(scores)
```

### Compare to the Default Classifier
Let's use the default `classification` problem type and compare the performance with the above.


```python
from autogluon.multimodal import MultiModalPredictor

predictor_default_text = MultiModalPredictor(
    label="label",
    problem_type="classification",
    eval_metric="acc",
)
predictor_default_text.fit(train_data=train_df)
scores = predictor_default_text.evaluate(test_df, metrics=["acc", "f1_macro"])
print(scores)
```

## Few Shot Image Classification
We also provide an example of using `MultiModalPredictor` on a few-shot image classification task.
### Load Dataset
We use the Stanford Cars dataset for demonstration and have downsampled the training set to have 8 samples per class.
The Stanford Cars is an image classification dataset and contains 196 classes.
For more information regarding the dataset, please see [here](https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset).


```python
import os
from autogluon.core.utils.loaders import load_zip, load_s3

download_dir = "./ag_automm_tutorial_fs_cls/stanfordcars/"
zip_file = "https://automl-mm-bench.s3.amazonaws.com/vision_datasets/stanfordcars/stanfordcars.zip"
train_csv = "https://automl-mm-bench.s3.amazonaws.com/vision_datasets/stanfordcars/train_8shot.csv"
test_csv = "https://automl-mm-bench.s3.amazonaws.com/vision_datasets/stanfordcars/test.csv"

load_zip.unzip(zip_file, unzip_dir=download_dir)
dataset_path = os.path.join(download_dir)

```


```python
!wget https://automl-mm-bench.s3.amazonaws.com/vision_datasets/stanfordcars/train_8shot.csv -O ./ag_automm_tutorial_fs_cls/stanfordcars/train.csv
!wget https://automl-mm-bench.s3.amazonaws.com/vision_datasets/stanfordcars/test.csv -O ./ag_automm_tutorial_fs_cls/stanfordcars/test.csv

```


```python
import pandas as pd
import os

train_df_raw = pd.read_csv(os.path.join(download_dir, "train.csv"))
train_df = train_df_raw.drop(
        columns=[
            "Source",
            "Confidence",
            "XMin",
            "XMax",
            "YMin",
            "YMax",
            "IsOccluded",
            "IsTruncated",
            "IsGroupOf",
            "IsDepiction",
            "IsInside",
        ]
    )
train_df["ImageID"] = download_dir + train_df["ImageID"].astype(str)


test_df_raw = pd.read_csv(os.path.join(download_dir, "test.csv"))
test_df = test_df_raw.drop(
        columns=[
            "Source",
            "Confidence",
            "XMin",
            "XMax",
            "YMin",
            "YMax",
            "IsOccluded",
            "IsTruncated",
            "IsGroupOf",
            "IsDepiction",
            "IsInside",
        ]
    )
test_df["ImageID"] = download_dir + test_df["ImageID"].astype(str)

print(os.path.exists(train_df.iloc[0]["ImageID"]))
print(train_df)
print(os.path.exists(test_df.iloc[0]["ImageID"]))
print(test_df)
```

### Train a Few Shot Classifier
Similarly, we need to initialize `MultiModalPredictor` with the problem type `few_shot_classification`.


```python
from autogluon.multimodal import MultiModalPredictor

predictor_fs_image = MultiModalPredictor(
    problem_type="few_shot_classification",
    label="LabelName",  # column name of the label
    eval_metric="acc",
)
predictor_fs_image.fit(train_df)
scores = predictor_fs_image.evaluate(test_df, metrics=["acc", "f1_macro"])
print(scores)
```

### Compare to the Default Classifier
We can also train a default image classifier and compare to the few shot classifier.


```python
from autogluon.multimodal import MultiModalPredictor

predictor_default_image = MultiModalPredictor(
    problem_type="classification",
    label="LabelName",  # column name of the label
    eval_metric="acc",
)
predictor_default_image.fit(train_data=train_df)
scores = predictor_default_image.evaluate(test_df, metrics=["acc", "f1_macro"])
print(scores)
```

As you can see that the `few_shot_classification` performs much better than the default `classification` in image classification as well.

## Customization
To learn how to customize AutoMM, please refer to [Customize AutoMM](../advanced_topics/customization.ipynb).
