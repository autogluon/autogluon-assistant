# Condensed: Few Shot Text Classification

Summary: This tutorial demonstrates few-shot classification implementation using AutoGluon's MultiModalPredictor for both text and image data. It covers techniques for training classifiers with limited samples by setting problem_type="few_shot_classification" versus standard classification approaches. Key functionalities include: preparing text/image datasets, configuring few-shot learning parameters, evaluating model performance with accuracy and F1 metrics, and comparing few-shot versus traditional classification approaches. The tutorial shows how to implement models that perform significantly better on limited training data, which is valuable for scenarios with scarce labeled examples.

*This is a condensed version that preserves essential implementation details and context.*

# Few Shot Text Classification

## Preparing Text Data

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
```

## Training a Few Shot Text Classifier

```python
from autogluon.multimodal import MultiModalPredictor

predictor_fs_text = MultiModalPredictor(
    problem_type="few_shot_classification",  # Key parameter for few-shot learning
    label="label",
    eval_metric="acc",
)
predictor_fs_text.fit(train_df)
scores = predictor_fs_text.evaluate(test_df, metrics=["acc", "f1_macro"])
print(scores)
```

## Comparison with Default Classifier

```python
predictor_default_text = MultiModalPredictor(
    label="label",
    problem_type="classification",  # Standard classification
    eval_metric="acc",
)
predictor_default_text.fit(train_data=train_df)
scores = predictor_default_text.evaluate(test_df, metrics=["acc", "f1_macro"])
print(scores)
```

# Few Shot Image Classification

## Loading Image Dataset

```python
import os
from autogluon.core.utils.loaders import load_zip

download_dir = "./ag_automm_tutorial_fs_cls/stanfordcars/"
zip_file = "https://automl-mm-bench.s3.amazonaws.com/vision_datasets/stanfordcars/stanfordcars.zip"
load_zip.unzip(zip_file, unzip_dir=download_dir)

# Download CSV files
!wget https://automl-mm-bench.s3.amazonaws.com/vision_datasets/stanfordcars/train_8shot.csv -O ./ag_automm_tutorial_fs_cls/stanfordcars/train.csv
!wget https://automl-mm-bench.s3.amazonaws.com/vision_datasets/stanfordcars/test.csv -O ./ag_automm_tutorial_fs_cls/stanfordcars/test.csv

# Process dataframes
import pandas as pd
train_df_raw = pd.read_csv(os.path.join(download_dir, "train.csv"))
train_df = train_df_raw.drop(
    columns=["Source", "Confidence", "XMin", "XMax", "YMin", "YMax", 
             "IsOccluded", "IsTruncated", "IsGroupOf", "IsDepiction", "IsInside"]
)
train_df["ImageID"] = download_dir + train_df["ImageID"].astype(str)

test_df_raw = pd.read_csv(os.path.join(download_dir, "test.csv"))
test_df = test_df_raw.drop(
    columns=["Source", "Confidence", "XMin", "XMax", "YMin", "YMax", 
             "IsOccluded", "IsTruncated", "IsGroupOf", "IsDepiction", "IsInside"]
)
test_df["ImageID"] = download_dir + test_df["ImageID"].astype(str)
```

## Training a Few Shot Image Classifier

```python
predictor_fs_image = MultiModalPredictor(
    problem_type="few_shot_classification",
    label="LabelName",
    eval_metric="acc",
)
predictor_fs_image.fit(train_df)
scores = predictor_fs_image.evaluate(test_df, metrics=["acc", "f1_macro"])
print(scores)
```

## Comparison with Default Image Classifier

```python
predictor_default_image = MultiModalPredictor(
    problem_type="classification",
    label="LabelName",
    eval_metric="acc",
)
predictor_default_image.fit(train_data=train_df)
scores = predictor_default_image.evaluate(test_df, metrics=["acc", "f1_macro"])
print(scores)
```

**Note**: The `few_shot_classification` problem type performs significantly better than the default `classification` for both text and image classification when working with limited training samples.