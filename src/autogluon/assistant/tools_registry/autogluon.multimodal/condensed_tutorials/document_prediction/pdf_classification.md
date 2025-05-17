# Condensed: Classifying PDF Documents with AutoMM

Summary: This tutorial demonstrates PDF document classification using AutoGluon's MultiModalPredictor, specifically implementing document processing and classification tasks. It covers essential techniques for handling PDF documents, including data preparation, model training with LayoutLM, and extraction of document embeddings. Key functionalities include automatic PDF processing, text recognition, probability predictions, and embedding extraction. The tutorial helps with tasks like setting up document paths, training classifiers, making predictions, and evaluating model performance. It provides implementation details for configuring the document transformer, managing training time limits, and handling PDF datasets effectively using AutoGluon's multimodal capabilities.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details:

# PDF Document Classification with AutoMM

## Prerequisites
- Requires `poppler` installation:
  - Windows: Install from [poppler-windows](https://github.com/oschwartz10612/poppler-windows) (add bin/ to PATH)
  - Mac: `brew install poppler`
  - Linux: Install from [poppler](https://poppler.freedesktop.org)

## Implementation Steps

### 1. Setup and Data Preparation
```python
!pip install autogluon.multimodal

import pandas as pd
from autogluon.core.utils.loaders import load_zip
from autogluon.multimodal.utils.misc import path_expander

# Download and prepare dataset
download_dir = './ag_automm_tutorial_pdf_classifier'
zip_file = "https://automl-mm-bench.s3.amazonaws.com/doc_classification/pdf_docs_small.zip"
load_zip.unzip(zip_file, unzip_dir=download_dir)

# Load and split data
dataset_path = os.path.join(download_dir, "pdf_docs_small")
pdf_docs = pd.read_csv(f"{dataset_path}/data.csv")
train_data = pdf_docs.sample(frac=0.8, random_state=200)
test_data = pdf_docs.drop(train_data.index)

# Update document paths
DOC_PATH_COL = "doc_path"
train_data[DOC_PATH_COL] = train_data[DOC_PATH_COL].apply(
    lambda ele: path_expander(ele, base_folder=download_dir)
)
test_data[DOC_PATH_COL] = test_data[DOC_PATH_COL].apply(
    lambda ele: path_expander(ele, base_folder=download_dir)
)
```

### 2. Create and Train Classifier
```python
from autogluon.multimodal import MultiModalPredictor

predictor = MultiModalPredictor(label="label")
predictor.fit(
    train_data=train_data,
    hyperparameters={
        "model.document_transformer.checkpoint_name": "microsoft/layoutlm-base-uncased",
        "optimization.top_k_average_method": "best",
    },
    time_limit=120,
)
```

### 3. Evaluation and Prediction
```python
# Evaluate
scores = predictor.evaluate(test_data, metrics=["accuracy"])

# Single prediction
predictions = predictor.predict({DOC_PATH_COL: [test_data.iloc[0][DOC_PATH_COL]]})

# Probability prediction
proba = predictor.predict_proba({DOC_PATH_COL: [test_data.iloc[0][DOC_PATH_COL]]})

# Extract embeddings
feature = predictor.extract_embedding({DOC_PATH_COL: [test_data.iloc[0][DOC_PATH_COL]]})
```

## Key Features
- Automatic PDF processing and format detection
- Built-in text recognition
- Document embedding extraction
- Support for probability predictions

## Important Configurations
- Uses `layoutlm-base-uncased` as default document transformer
- Time limit can be adjusted based on dataset size and requirements
- Supports various evaluation metrics

## Best Practices
1. Ensure correct document paths are provided
2. Verify PDF accessibility before processing
3. Consider memory requirements for large PDF datasets
4. Use appropriate time limits based on dataset size

For customization options, refer to the AutoMM customization documentation.