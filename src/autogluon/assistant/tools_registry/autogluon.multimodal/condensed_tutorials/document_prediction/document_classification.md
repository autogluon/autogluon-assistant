# Condensed: Get a Document Dataset

Summary: This tutorial demonstrates document classification using AutoGluon MultiModal, showing how to implement a document classifier that recognizes text, layout, and visual features in documents. It covers loading document data, training a model using LayoutLM architecture, evaluating performance, and extracting embeddings. The tutorial shows how to leverage pre-trained document foundation models for classification tasks without requiring extensive feature engineering. Key functionalities include automatic feature recognition, support for various document models, and the ability to extract embeddings for downstream tasks. The implementation uses MultiModalPredictor to simplify the document classification workflow with minimal code.

*This is a condensed version that preserves essential implementation details and context.*

# Document Classification with AutoGluon MultiModal

## Dataset Setup

```python
!pip install autogluon.multimodal

import warnings
warnings.filterwarnings('ignore')
import os
import pandas as pd
from autogluon.core.utils.loaders import load_zip

# Download and extract dataset
download_dir = './ag_automm_tutorial_doc_classifier'
zip_file = "https://automl-mm-bench.s3.amazonaws.com/doc_classification/rvl_cdip_sample.zip"
load_zip.unzip(zip_file, unzip_dir=download_dir)

# Load and split data
dataset_path = os.path.join(download_dir, "rvl_cdip_sample")
rvl_cdip_data = pd.read_csv(f"{dataset_path}/rvl_cdip_train_data.csv")
train_data = rvl_cdip_data.sample(frac=0.8, random_state=200)
test_data = rvl_cdip_data.drop(train_data.index)

# Expand document paths
from autogluon.multimodal.utils.misc import path_expander
DOC_PATH_COL = "doc_path"
train_data[DOC_PATH_COL] = train_data[DOC_PATH_COL].apply(lambda ele: path_expander(ele, base_folder=download_dir))
test_data[DOC_PATH_COL] = test_data[DOC_PATH_COL].apply(lambda ele: path_expander(ele, base_folder=download_dir))
```

## Building the Document Classifier

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

**Key Implementation Details:**
- AutoMM automatically recognizes text, layout information, and visual features for document classification
- Supports document foundation models like layoutlmv3, layoutlmv2, layoutlm-base, layoutxlm
- Also works with pure text models like bert, deberta

## Evaluation and Prediction

```python
# Evaluate on test data
scores = predictor.evaluate(test_data, metrics=["accuracy"])
print('The test acc: %.3f' % scores["accuracy"])

# Predict on a new document
doc_path = test_data.iloc[1][DOC_PATH_COL]
predictions = predictor.predict({DOC_PATH_COL: [doc_path]})

# Get prediction probabilities
proba = predictor.predict_proba({DOC_PATH_COL: [doc_path]})
```

## Feature Extraction

```python
# Extract document embeddings
feature = predictor.extract_embedding({DOC_PATH_COL: [doc_path]})
print(feature[0].shape)  # Shape depends on the model used
```

**Note:** For customization options, refer to the "Customize AutoMM" documentation. Additional examples are available in the AutoMM Examples repository.