Summary: This tutorial provides implementation guidance for AutoMM's diverse problem types and metrics, covering classification (binary/multiclass), regression, computer vision (object detection, semantic segmentation), similarity matching, NLP tasks (NER), and feature extraction. It helps with tasks involving metric selection, modality support checking, and zero-shot prediction implementation. Key features include detailed metric configurations for each problem type, modality support specifications (text, image, numerical, categorical), zero-shot capability identification, and access to the PROBLEM_TYPES_REG registry for customization. The tutorial is particularly valuable for implementing multi-modal machine learning solutions and understanding which metrics and modalities are supported for different problem types.

# AutoMM Problem Types And Metrics

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/autogluon/autogluon/blob/master/docs/tutorials/multimodal/advanced_topics/problem_types_and_metrics.ipynb)
[![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/autogluon/autogluon/blob/master/docs/tutorials/multimodal/advanced_topics/problem_types_and_metrics.ipynb)


AutoGluon Multimodal supports various problem types for different machine learning tasks. In this tutorial, we will introduce each problem type, its supported modalities, and evaluation metrics.


```python
!pip install autogluon.multimodal
```


```python
import warnings

warnings.filterwarnings('ignore')
```

Lets first write a helper function to print problem type information in a formatted way.


```python
from autogluon.multimodal.constants import *
from autogluon.multimodal.problem_types import PROBLEM_TYPES_REG

def print_problem_type_info(name: str, props):
    """Helper function to print problem type information"""
    print(f"\n=== {name} ===")
    
    print("\nSupported Input Modalities:")
    # Convert set to sorted list for complete display
    for modality in sorted(list(props.supported_modality_type)):
        print(f"- {modality}")
        
    if hasattr(props, 'supported_evaluation_metrics') and props.supported_evaluation_metrics:
        print("\nEvaluation Metrics:")
        # Convert to sorted list to ensure complete and consistent display
        for metric in sorted(list(props.supported_evaluation_metrics)):
            if metric == props.fallback_evaluation_metric:
                print(f"- {metric} (default)")
            else:
                print(f"- {metric}")
                
    if hasattr(props, 'support_zero_shot'):
        print("\nSpecial Capabilities:")
        print(f"- Zero-shot prediction: {'Supported' if props.support_zero_shot else 'Not supported'}")
        print(f"- Training support: {'Supported' if props.support_fit else 'Not supported'}")
```

## Classification

AutoGluon supports two types of classification:

- Binary Classification (2 classes)
- Multiclass Classification (3+ classes)


```python
# Classification
binary_props = PROBLEM_TYPES_REG.get(BINARY)
multiclass_props = PROBLEM_TYPES_REG.get(MULTICLASS)
print_problem_type_info("Binary Classification", binary_props)
print_problem_type_info("Multiclass Classification", multiclass_props)
```

    
    === Binary Classification ===
    
    Supported Input Modalities:
    - categorical
    - image
    - image_base64_str
    - image_bytearray
    - numerical
    - text
    
    Evaluation Metrics:
    - acc
    - accuracy
    - average_precision
    - balanced_accuracy
    - f1
    - f1_macro
    - f1_micro
    - f1_weighted
    - log_loss
    - mcc
    - nll
    - pac
    - pac_score
    - precision
    - precision_macro
    - precision_micro
    - precision_weighted
    - quadratic_kappa
    - recall
    - recall_macro
    - recall_micro
    - recall_weighted
    - roc_auc (default)
    
    Special Capabilities:
    - Zero-shot prediction: Not supported
    - Training support: Supported
    
    === Multiclass Classification ===
    
    Supported Input Modalities:
    - categorical
    - image
    - image_base64_str
    - image_bytearray
    - numerical
    - text
    
    Evaluation Metrics:
    - acc
    - accuracy (default)
    - balanced_accuracy
    - f1_macro
    - f1_micro
    - f1_weighted
    - log_loss
    - mcc
    - nll
    - pac
    - pac_score
    - precision_macro
    - precision_micro
    - precision_weighted
    - quadratic_kappa
    - recall_macro
    - recall_micro
    - recall_weighted
    - roc_auc_ovo
    - roc_auc_ovo_macro
    - roc_auc_ovo_weighted
    - roc_auc_ovr
    - roc_auc_ovr_macro
    - roc_auc_ovr_micro
    - roc_auc_ovr_weighted
    
    Special Capabilities:
    - Zero-shot prediction: Not supported
    - Training support: Supported


## Regression

Regression problems support predicting numerical values from various input modalities.


```python
# Regression
regression_props = PROBLEM_TYPES_REG.get(REGRESSION)
print_problem_type_info("Regression", regression_props)
```

    
    === Regression ===
    
    Supported Input Modalities:
    - categorical
    - image
    - image_base64_str
    - image_bytearray
    - numerical
    - text
    
    Evaluation Metrics:
    - mae
    - mape
    - mean_absolute_error
    - mean_absolute_percentage_error
    - mean_squared_error
    - median_absolute_error
    - mse
    - pearsonr
    - r2
    - rmse (default)
    - root_mean_squared_error
    - smape
    - spearmanr
    - symmetric_mean_absolute_percentage_error
    
    Special Capabilities:
    - Zero-shot prediction: Not supported
    - Training support: Supported


## Object Detection

Object detection identifies and localizes objects in images using bounding boxes.


```python
# Object Detection
object_detection_props = PROBLEM_TYPES_REG.get(OBJECT_DETECTION)
print_problem_type_info("Object Detection", object_detection_props)
```

    
    === Object Detection ===
    
    Supported Input Modalities:
    - image
    
    Evaluation Metrics:
    - map (default)
    - map_50
    - map_75
    - map_large
    - map_medium
    - map_small
    - mar_1
    - mar_10
    - mar_100
    - mar_large
    - mar_medium
    - mar_small
    - mean_average_precision
    
    Special Capabilities:
    - Zero-shot prediction: Supported
    - Training support: Supported


## Semantic Segmentation

Semantic segmentation performs pixel-level classification of images.


```python
# Semantic Segmentation
segmentation_props = PROBLEM_TYPES_REG.get(SEMANTIC_SEGMENTATION)
print_problem_type_info("Semantic Segmentation", segmentation_props)
```

    
    === Semantic Segmentation ===
    
    Supported Input Modalities:
    - image
    
    Evaluation Metrics:
    - ber
    - iou (default)
    - sm
    
    Special Capabilities:
    - Zero-shot prediction: Supported
    - Training support: Supported


## Similarity Matching Problems

AutoGluon supports three types of similarity matching:

- Text-to-Text Similarity
- Image-to-Image Similarity
- Image-to-Text Similarity

Check [Matching Tutorials](../semantic_matching/index.md) for more details


```python
similarity_types = [
    (TEXT_SIMILARITY, "Text Similarity"),
    (IMAGE_SIMILARITY, "Image Similarity"),
    (IMAGE_TEXT_SIMILARITY, "Image-Text Similarity")
]

print("\n=== Similarity Matching ===")
for type_key, type_name in similarity_types:
    props = PROBLEM_TYPES_REG.get(type_key)
    print(f"\n{type_name}:")
    print("Input Requirements:")
    for modality in props.supported_modality_type:
        print(f"- {modality}")
    print(f"Zero-shot prediction: {'Supported' if props.support_zero_shot else 'Not supported'}")

```

    
    === Similarity Matching ===
    
    Text Similarity:
    Input Requirements:
    - text
    Zero-shot prediction: Supported
    
    Image Similarity:
    Input Requirements:
    - image
    Zero-shot prediction: Supported
    
    Image-Text Similarity:
    Input Requirements:
    - text
    - image
    Zero-shot prediction: Supported


## Named Entity Recognition (NER)

NER identifies and classifies named entities (like person names, locations, organizations) in text.


```python
# Named Entity Recognition
ner_props = PROBLEM_TYPES_REG.get(NER)
print_problem_type_info("Named Entity Recognition", ner_props)
```

    
    === Named Entity Recognition ===
    
    Supported Input Modalities:
    - categorical
    - image
    - numerical
    - text
    - text_ner
    
    Evaluation Metrics:
    - ner_token_f1
    - overall_f1 (default)
    
    Special Capabilities:
    - Zero-shot prediction: Not supported
    - Training support: Supported


## Feature Extraction

Feature extraction converts raw data into meaningful feature vector.


```python
# Feature Extraction
feature_extraction_props = PROBLEM_TYPES_REG.get(FEATURE_EXTRACTION)
print_problem_type_info("Feature Extraction", feature_extraction_props)
```

    
    === Feature Extraction ===
    
    Supported Input Modalities:
    - image
    - text
    
    Special Capabilities:
    - Zero-shot prediction: Supported
    - Training support: Not supported


## Few-shot Classification

Few-shot classification learns to classify from a small number of examples per class.


```python
# Few-shot Classification
few_shot_props = PROBLEM_TYPES_REG.get(FEW_SHOT_CLASSIFICATION)
print_problem_type_info("Few-shot Classification", few_shot_props)
```

    
    === Few-shot Classification ===
    
    Supported Input Modalities:
    - image
    - text
    
    Evaluation Metrics:
    - acc
    - accuracy (default)
    - balanced_accuracy
    - f1_macro
    - f1_micro
    - f1_weighted
    - log_loss
    - mcc
    - nll
    - pac
    - pac_score
    - precision_macro
    - precision_micro
    - precision_weighted
    - quadratic_kappa
    - recall_macro
    - recall_micro
    - recall_weighted
    - roc_auc_ovo
    - roc_auc_ovo_macro
    - roc_auc_ovo_weighted
    - roc_auc_ovr
    - roc_auc_ovr_macro
    - roc_auc_ovr_micro
    - roc_auc_ovr_weighted
    
    Special Capabilities:
    - Zero-shot prediction: Not supported
    - Training support: Supported


## Other Examples
You may go to [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm) to explore other examples about AutoMM.

## Customization
To learn how to customize AutoMM, please refer to [Customize AutoMM](../advanced_topics/customization.ipynb).
