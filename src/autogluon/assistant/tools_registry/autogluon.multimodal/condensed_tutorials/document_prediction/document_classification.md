# Condensed: Get a Document Dataset

Summary: This tutorial demonstrates building scanned document classifiers using AutoGluon's MultiModalPredictor with document transformer models (LayoutLM, LayoutLMv2/v3, LayoutXLM, BERT, DeBERTa). It covers essential implementation techniques including dataset preparation with path expansion using `path_expander`, model training with customizable hyperparameters (`model.document_transformer.checkpoint_name`), and inference operations (predict, predict_proba, extract_embedding). Key functionalities include automatic text recognition (handwritten/typed), layout analysis, visual feature extraction, model evaluation with metrics, probability prediction for multi-class classification, and document embedding extraction. The tutorial helps with tasks like document classification pipeline setup, model selection and configuration, data preprocessing for document images, and deploying pre-trained transformer models for document understanding applications.

*This is a condensed version that preserves essential implementation details and context.*

# Document Classification with AutoMM

## Dataset Setup

Download and prepare the RVL-CDIP sample dataset (100 documents, 3 categories: budget=0, email=1, form=2):

```python
!pip install autogluon.multimodal

import warnings
warnings.filterwarnings('ignore')
import os
import pandas as pd
from autogluon.core.utils.loaders import load_zip

download_dir = './ag_automm_tutorial_doc_classifier'
zip_file = "https://automl-mm-bench.s3.amazonaws.com/doc_classification/rvl_cdip_sample.zip"
load_zip.unzip(zip_file, unzip_dir=download_dir)

# Load and split data
dataset_path = os.path.join(download_dir, "rvl_cdip_sample")
rvl_cdip_data = pd.read_csv(f"{dataset_path}/rvl_cdip_train_data.csv")
train_data = rvl_cdip_data.sample(frac=0.8, random_state=200)
test_data = rvl_cdip_data.drop(train_data.index)
```

**Critical step:** Expand document paths before training:

```python
from autogluon.multimodal.utils.misc import path_expander

DOC_PATH_COL = "doc_path"
train_data[DOC_PATH_COL] = train_data[DOC_PATH_COL].apply(
    lambda ele: path_expander(ele, base_folder=download_dir))
test_data[DOC_PATH_COL] = test_data[DOC_PATH_COL].apply(
    lambda ele: path_expander(ele, base_folder=download_dir))
```

## Build Document Classifier

AutoMM automatically recognizes text (handwritten/typed), layout, and visual features:

```python
from autogluon.multimodal import MultiModalPredictor

predictor = MultiModalPredictor(label="label")
predictor.fit(
    train_data=train_data,
    hyperparameters={
        "model.document_transformer.checkpoint_name": "microsoft/layoutlm-base-uncased",
        "optim.top_k_average_method": "best",
    },
    time_limit=120,
)
```

**Supported models:** layoutlmv3, layoutlmv2, layoutlm-base, layoutxlm, bert, deberta

## Inference

**Evaluate:**
```python
scores = predictor.evaluate(test_data, metrics=["accuracy"])
print('The test acc: %.3f' % scores["accuracy"])
```

**Predict class:**
```python
predictions = predictor.predict({DOC_PATH_COL: [doc_path]})
```

**Predict probabilities:**
```python
proba = predictor.predict_proba({DOC_PATH_COL: [doc_path]})
```

**Extract embeddings:**
```python
feature = predictor.extract_embedding({DOC_PATH_COL: [doc_path]})
print(feature[0].shape)  # N-dimensional feature (N depends on model)
```

## Additional Resources
- [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm)
- [Customize AutoMM](../advanced_topics/customization.ipynb)