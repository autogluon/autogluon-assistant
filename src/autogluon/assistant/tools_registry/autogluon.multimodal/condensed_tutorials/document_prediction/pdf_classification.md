# Condensed: Get the PDF document dataset

Summary: "Summarize.md"

Summary: "Summarize.md"

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: Summary: This tutorial demonstrates how.

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: This tutorial demonstrates PDF document classification using AutoGluon's MultiModalPredictor. It covers implementing a complete workflow for document classification including dataset preparation, model training with LayoutLM, evaluation, prediction, and embedding extraction. The tutorial shows how AutoGluon handles PDF processing automatically, requiring minimal code to build a document classifier. Key functionalities include proper path configuration for PDF files, model training with time constraints, accuracy evaluation, and extracting document embeddings for downstream tasks.

*This is a condensed version that preserves essential implementation details and context.*

# PDF Document Classification with AutoGluon

## Dataset Setup

```python
!pip install autogluon.multimodal

import warnings
warnings.filterwarnings('ignore')
import os
import pandas as pd
from autogluon.core.utils.loaders import load_zip

# Download and extract dataset
download_dir = './ag_automm_tutorial_pdf_classifier'
zip_file = "https://automl-mm-bench.s3.amazonaws.com/doc_classification/pdf_docs_small.zip"
load_zip.unzip(zip_file, unzip_dir=download_dir)

# Load and split dataset
dataset_path = os.path.join(download_dir, "pdf_docs_small")
pdf_docs = pd.read_csv(f"{dataset_path}/data.csv")
train_data = pdf_docs.sample(frac=0.8, random_state=200)
test_data = pdf_docs.drop(train_data.index)

# Fix document paths
from autogluon.multimodal.utils.misc import path_expander
DOC_PATH_COL = "doc_path"
train_data[DOC_PATH_COL] = train_data[DOC_PATH_COL].apply(lambda ele: path_expander(ele, base_folder=download_dir))
test_data[DOC_PATH_COL] = test_data[DOC_PATH_COL].apply(lambda ele: path_expander(ele, base_folder=download_dir))
```

## Create and Train PDF Classifier

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

## Evaluate and Predict

```python
# Evaluate on test data
scores = predictor.evaluate(test_data, metrics=["accuracy"])
print('The test acc: %.3f' % scores["accuracy"])

# Make predictions
predictions = predictor.predict({DOC_PATH_COL: [test_data.iloc[0][DOC_PATH_COL]]})
print(f"Ground-truth label: {test_data.iloc[0]['label']}, Prediction: {predictions}")

# Get prediction probabilities
proba = predictor.predict_proba({DOC_PATH_COL: [test_data.iloc[0][DOC_PATH_COL]]})
print(proba)
```

## Extract Embeddings

```python
# Extract document embeddings
feature = predictor.extract_embedding({DOC_PATH_COL: [test_data.iloc[0][DOC_PATH_COL]]})
print(feature[0].shape)
```

## Key Implementation Details

- AutoGluon automatically handles PDF processing, including format detection and text recognition
- Uses LayoutLM model for document understanding
- The document path column must be properly configured to locate PDF files
- Supports standard operations: training, evaluation, prediction, and embedding extraction

For customization options, refer to the AutoMM customization documentation.