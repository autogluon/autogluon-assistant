# Condensed: Few Shot Text Classification

Summary: This tutorial demonstrates implementing few-shot learning for text and image classification using AutoGluon's MultiModalPredictor with the critical `problem_type="few_shot_classification"` parameter. It covers: (1) preparing DataFrame-formatted datasets from MLDoc (text, 10-shot) and Stanford Cars (images, 8-shot), (2) training few-shot classifiers with label specification and evaluation metrics, (3) preprocessing image data by dropping metadata columns and fixing file paths, and (4) comparing few-shot vs. default classification performance. Key implementation details include using `load_zip.unzip()` for dataset loading, configuring `MultiModalPredictor` with `label` and `eval_metric` parameters, and evaluating with multiple metrics (`acc`, `f1_macro`). Useful for building classifiers with limited training samples across text and vision domains.

*This is a condensed version that preserves essential implementation details and context.*

# Few Shot Text Classification

## Prepare Text Data
Uses MLDoc dataset (4 classes, 10 samples per class). Data formatted as `pd.DataFrame`.

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

## Train Few Shot Text Classifier
**Key**: Use `problem_type="few_shot_classification"`

```python
from autogluon.multimodal import MultiModalPredictor

predictor_fs_text = MultiModalPredictor(
    problem_type="few_shot_classification",
    label="label",
    eval_metric="acc",
)
predictor_fs_text.fit(train_df)
scores = predictor_fs_text.evaluate(test_df, metrics=["acc", "f1_macro"])
```

## Comparison with Default Classifier

```python
predictor_default_text = MultiModalPredictor(
    label="label",
    problem_type="classification",
    eval_metric="acc",
)
predictor_default_text.fit(train_data=train_df)
scores = predictor_default_text.evaluate(test_df, metrics=["acc", "f1_macro"])
```

# Few Shot Image Classification

## Load Dataset
Stanford Cars dataset (196 classes, 8 samples per class).

```python
from autogluon.core.utils.loaders import load_zip

download_dir = "./ag_automm_tutorial_fs_cls/stanfordcars/"
zip_file = "https://automl-mm-bench.s3.amazonaws.com/vision_datasets/stanfordcars/stanfordcars.zip"
load_zip.unzip(zip_file, unzip_dir=download_dir)
```

```bash
wget https://automl-mm-bench.s3.amazonaws.com/vision_datasets/stanfordcars/train_8shot.csv -O ./ag_automm_tutorial_fs_cls/stanfordcars/train.csv
wget https://automl-mm-bench.s3.amazonaws.com/vision_datasets/stanfordcars/test.csv -O ./ag_automm_tutorial_fs_cls/stanfordcars/test.csv
```

**Data preprocessing**: Drop metadata columns, prepend directory path to ImageID.

```python
train_df = pd.read_csv(os.path.join(download_dir, "train.csv"))
train_df = train_df.drop(columns=["Source", "Confidence", "XMin", "XMax", "YMin", "YMax", 
                                   "IsOccluded", "IsTruncated", "IsGroupOf", "IsDepiction", "IsInside"])
train_df["ImageID"] = download_dir + train_df["ImageID"].astype(str)
# Repeat for test_df
```

## Train Few Shot Image Classifier

```python
predictor_fs_image = MultiModalPredictor(
    problem_type="few_shot_classification",
    label="LabelName",
    eval_metric="acc",
)
predictor_fs_image.fit(train_df)
scores = predictor_fs_image.evaluate(test_df, metrics=["acc", "f1_macro"])
```

## Comparison with Default Classifier

```python
predictor_default_image = MultiModalPredictor(
    problem_type="classification",
    label="LabelName",
    eval_metric="acc",
)
predictor_default_image.fit(train_data=train_df)
scores = predictor_default_image.evaluate(test_df, metrics=["acc", "f1_macro"])
```

**Result**: `few_shot_classification` significantly outperforms default `classification` for both text and image tasks.

## Customization
See [Customize AutoMM](../advanced_topics/customization.ipynb) for advanced configurations.