# Condensed: AutoMM for Scanned Document Classification

Summary: This tutorial demonstrates implementing document classification using AutoGluon's MultiModal Predictor, specifically focusing on scanned document processing. It covers techniques for handling document datasets, training layout-aware models (like LayoutLM variants), and performing predictions/feature extraction. The tutorial helps with tasks including document classification setup, model training configuration, and inference pipeline implementation. Key functionalities include automatic text recognition, layout information processing, support for multiple document/text models (LayoutLM, BERT, DeBERTa), and flexible model customization through hyperparameters. The implementation provides a streamlined approach to building document classification systems with minimal code while maintaining adaptability.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details:

# AutoMM for Scanned Document Classification

## Key Implementation Details

### Setup and Data Preparation
```python
# Install required package
pip install autogluon.multimodal

# Load and prepare data
import pandas as pd
from autogluon.core.utils.loaders import load_zip
from autogluon.multimodal.utils.misc import path_expander

# Download and unzip dataset
download_dir = './ag_automm_tutorial_doc_classifier'
zip_file = "https://automl-mm-bench.s3.amazonaws.com/doc_classification/rvl_cdip_sample.zip"
load_zip.unzip(zip_file, unzip_dir=download_dir)

# Split data
dataset_path = os.path.join(download_dir, "rvl_cdip_sample")
rvl_cdip_data = pd.read_csv(f"{dataset_path}/rvl_cdip_train_data.csv")
train_data = rvl_cdip_data.sample(frac=0.8, random_state=200)
test_data = rvl_cdip_data.drop(train_data.index)

# Expand document paths
DOC_PATH_COL = "doc_path"
train_data[DOC_PATH_COL] = train_data[DOC_PATH_COL].apply(lambda ele: path_expander(ele, base_folder=download_dir))
test_data[DOC_PATH_COL] = test_data[DOC_PATH_COL].apply(lambda ele: path_expander(ele, base_folder=download_dir))
```

### Model Training and Prediction
```python
from autogluon.multimodal import MultiModalPredictor

# Initialize and train predictor
predictor = MultiModalPredictor(label="label")
predictor.fit(
    train_data=train_data,
    hyperparameters={
        "model.document_transformer.checkpoint_name": "microsoft/layoutlm-base-uncased",
        "optimization.top_k_average_method": "best",
    },
    time_limit=120,
)

# Evaluate
scores = predictor.evaluate(test_data, metrics=["accuracy"])

# Make predictions
predictions = predictor.predict({DOC_PATH_COL: [doc_path]})
probabilities = predictor.predict_proba({DOC_PATH_COL: [doc_path]})

# Extract embeddings
feature = predictor.extract_embedding({DOC_PATH_COL: [doc_path]})
```

## Important Notes

1. **Supported Models**:
   - Document models: layoutlmv3, layoutlmv2, layoutlm-base, layoutxlm
   - Text models: bert, deberta

2. **Key Features**:
   - Automatic recognition of handwritten/typed text
   - Utilizes text, layout information, and visual features
   - Easy model customization through hyperparameters

3. **Dataset Structure**:
   - Sample from RVL-CDIP dataset
   - 3 categories: budget (0), email (1), form (2)
   - Uses grayscale images

4. **Best Practices**:
   - Expand document paths before training
   - Set appropriate time limits based on dataset size
   - Use evaluation metrics to assess model performance

This implementation allows for quick deployment of document classification systems with minimal code while maintaining flexibility for customization.