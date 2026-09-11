# Condensed: Get the PDF document dataset

Summary: This tutorial demonstrates PDF document classification using AutoGluon's MultiModalPredictor with LayoutLM. It covers: (1) dataset preparation with train/test splitting and critical path expansion using `path_expander`, (2) training a classifier with hyperparameter configuration including LayoutLM checkpoint selection and optimization methods, (3) model evaluation with accuracy metrics, (4) inference capabilities including class prediction, probability estimation, and document embedding extraction. Key implementation knowledge includes handling PDF file paths in pandas DataFrames, configuring document transformer models, and utilizing AutoMM's automatic PDF processing pipeline. This tutorial helps with tasks like building PDF classifiers for document categorization (resumes, contracts, forms), implementing document understanding systems, and extracting semantic features from PDFs for downstream applications.

*This is a condensed version that preserves essential implementation details and context.*

## PDF Document Classification with AutoMM

### Dataset Setup
Dataset: 40 PDFs (20 resumes + 20 historical documents from [milestone documents](https://www.archives.gov/milestone-documents/list))

```python
!pip install autogluon.multimodal

import warnings
warnings.filterwarnings('ignore')
import os
import pandas as pd
from autogluon.core.utils.loaders import load_zip

download_dir = './ag_automm_tutorial_pdf_classifier'
zip_file = "https://automl-mm-bench.s3.amazonaws.com/doc_classification/pdf_docs_small.zip"
load_zip.unzip(zip_file, unzip_dir=download_dir)

dataset_path = os.path.join(download_dir, "pdf_docs_small")
pdf_docs = pd.read_csv(f"{dataset_path}/data.csv")
train_data = pdf_docs.sample(frac=0.8, random_state=200)
test_data = pdf_docs.drop(train_data.index)
```

**Critical**: Update document paths using `path_expander`:

```python
from autogluon.multimodal.utils.misc import path_expander

DOC_PATH_COL = "doc_path"
train_data[DOC_PATH_COL] = train_data[DOC_PATH_COL].apply(lambda ele: path_expander(ele, base_folder=download_dir))
test_data[DOC_PATH_COL] = test_data[DOC_PATH_COL].apply(lambda ele: path_expander(ele, base_folder=download_dir))
```

### Training

AutoMM automatically handles: (1) PDF format detection, (2) PDF processing/conversion, (3) text detection and recognition.

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

**Key Parameters**:
- `label`: Target column name
- `checkpoint_name`: Uses LayoutLM base model
- `time_limit`: Training duration (120s for demo)

### Evaluation & Prediction

```python
# Evaluate
scores = predictor.evaluate(test_data, metrics=["accuracy"])

# Predict class
predictions = predictor.predict({DOC_PATH_COL: [test_data.iloc[0][DOC_PATH_COL]]})

# Predict probabilities
proba = predictor.predict_proba({DOC_PATH_COL: [test_data.iloc[0][DOC_PATH_COL]]})

# Extract embeddings
feature = predictor.extract_embedding({DOC_PATH_COL: [test_data.iloc[0][DOC_PATH_COL]]})
```

### Additional Resources
- [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm)
- [Customize AutoMM](../advanced_topics/customization.ipynb)