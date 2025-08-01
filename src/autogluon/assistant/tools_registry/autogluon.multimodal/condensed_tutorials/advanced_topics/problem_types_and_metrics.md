# Condensed: ```python

Summary: This tutorial explains AutoGluon MultiModal's problem types for code implementation, covering classification (binary and multiclass), regression, object detection, semantic segmentation, similarity matching, NER, and feature extraction. It helps with implementing models that handle various data modalities (text, image, numerical, categorical) and specifies which problem types support training vs. zero-shot prediction. The tutorial details evaluation metrics for each problem type and provides implementation context for working with multimodal data. It's valuable for coding tasks involving multimodal machine learning, particularly when working with mixed data types and needing automated model training across different problem domains.

*This is a condensed version that preserves essential implementation details and context.*

# AutoGluon MultiModal Problem Types

## Installation and Setup

```python
!pip install autogluon.multimodal
import warnings
warnings.filterwarnings('ignore')
```

## Problem Types Overview

### Classification

AutoGluon supports two types of classification:

**Binary Classification (2 classes)**
- Supported modalities: image, numerical, text, categorical, tabular
- Evaluation metrics: accuracy, balanced_accuracy, f1, mcc, roc_auc (default)
- Supports training and zero-shot prediction

**Multiclass Classification (3+ classes)**
- Supported modalities: image, numerical, text, categorical, tabular
- Evaluation metrics: accuracy (default), balanced_accuracy, f1, log_loss
- Supports training and zero-shot prediction

### Regression

- Supported modalities: image, numerical, text, categorical, tabular
- Evaluation metrics: mse, rmse (default), r2, mae, mape
- Supports training and zero-shot prediction

### Object Detection

- Supported modalities: image
- Evaluation metrics: map (default)
- Supports training but not zero-shot prediction

### Semantic Segmentation

- Supported modalities: image
- Evaluation metrics: overall_accuracy, mean_accuracy, mean_iou (default), weighted_iou
- Supports training but not zero-shot prediction

### Similarity Matching Problems

**Text Similarity**
- Input requirements: text
- Supports zero-shot prediction

**Image Similarity**
- Input requirements: image
- Supports zero-shot prediction

**Image-Text Similarity**
- Input requirements: image, text
- Supports zero-shot prediction

### Named Entity Recognition (NER)

- Supported modalities: text
- Evaluation metrics: overall_f1 (default)
- Supports training but not zero-shot prediction

### Feature Extraction

- Supported modalities: image, numerical, text, categorical, tabular
- Supports training and zero-shot prediction

### Few-shot Classification

- Supported modalities: image, text
- Evaluation metrics: accuracy (default)
- Supports training and zero-shot prediction

## Additional Resources
- For more examples: [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm)
- For customization: [Customize AutoMM](../advanced_topics/customization.ipynb)
- For similarity matching: [Matching Tutorials](../semantic_matching/index.md)