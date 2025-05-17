# Condensed: AutoMM Problem Types And Metrics

Summary: This tutorial provides implementation guidance for AutoMM's diverse problem types and metrics, covering classification (binary/multiclass), regression, computer vision (object detection, semantic segmentation), similarity matching, NLP tasks (NER), and feature extraction. It helps with tasks involving metric selection, modality support checking, and zero-shot prediction implementation. Key features include detailed metric configurations for each problem type, modality support specifications (text, image, numerical, categorical), zero-shot capability identification, and access to the PROBLEM_TYPES_REG registry for customization. The tutorial is particularly valuable for implementing multi-modal machine learning solutions and understanding which metrics and modalities are supported for different problem types.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details and key concepts:

# AutoMM Problem Types and Metrics Guide

## Core Problem Types

### 1. Classification
- **Binary Classification**
  - Input: categorical, numerical, text, image (including base64/bytearray)
  - Default metric: roc_auc
  - Key metrics: accuracy, f1, precision, recall, log_loss

- **Multiclass Classification**
  - Same input types as binary
  - Default metric: accuracy
  - Additional metrics: roc_auc_ovo, roc_auc_ovr variants

### 2. Regression
- Supports all standard modalities
- Default metric: rmse
- Key metrics: mae, mse, r2, pearsonr, spearmanr

### 3. Computer Vision Tasks
- **Object Detection**
  ```python
  # Input modality: image only
  # Default metric: map (mean average precision)
  # Supports zero-shot prediction
  ```

- **Semantic Segmentation**
  ```python
  # Input modality: image only
  # Default metric: iou
  # Supports zero-shot prediction
  ```

### 4. Similarity Matching
```python
# Three types:
- Text-to-Text Similarity
- Image-to-Image Similarity 
- Image-to-Text Similarity

# All support zero-shot prediction
```

### 5. NLP Tasks
- **Named Entity Recognition (NER)**
  - Input: text, text_ner, categorical, numerical, image
  - Default metric: overall_f1
  - Additional metric: ner_token_f1

### 6. Feature Extraction
```python
# Supported modalities:
- image
- text

# Key characteristics:
- Supports zero-shot prediction
- No training support (inference only)
```

### 7. Few-shot Classification
- Supports image and text modalities
- Default metric: accuracy
- Same evaluation metrics as multiclass classification

## Important Implementation Notes

1. Helper function for checking problem type details:
```python
from autogluon.multimodal.constants import *
from autogluon.multimodal.problem_types import PROBLEM_TYPES_REG

def print_problem_type_info(name: str, props):
    # Access supported modalities, metrics, and capabilities
    print(f"\nSupported Input Modalities:")
    for modality in sorted(list(props.supported_modality_type)):
        print(f"- {modality}")
    # ... additional property checks
```

2. Zero-shot capabilities vary by problem type:
   - Supported: Object Detection, Segmentation, Similarity Matching, Feature Extraction
   - Not supported: Classification, Regression, NER, Few-shot Classification

3. For customization options, refer to the Customize AutoMM documentation

4. All metrics and configurations can be accessed through the PROBLEM_TYPES_REG registry