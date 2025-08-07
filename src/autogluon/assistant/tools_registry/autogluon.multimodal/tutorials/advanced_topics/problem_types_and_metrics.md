Summary: This tutorial explains AutoGluon MultiModal's problem types for code implementation, covering classification (binary and multiclass), regression, object detection, semantic segmentation, similarity matching, NER, and feature extraction. It helps with implementing models that handle various data modalities (text, image, numerical, categorical) and specifies which problem types support training vs. zero-shot prediction. The tutorial details evaluation metrics for each problem type and provides implementation context for working with multimodal data. It's valuable for coding tasks involving multimodal machine learning, particularly when working with mixed data types and needing automated model training across different problem domains.

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

## Regression

Regression problems support predicting numerical values from various input modalities.


```python
# Regression
regression_props = PROBLEM_TYPES_REG.get(REGRESSION)
print_problem_type_info("Regression", regression_props)
```

## Object Detection

Object detection identifies and localizes objects in images using bounding boxes.


```python
# Object Detection
object_detection_props = PROBLEM_TYPES_REG.get(OBJECT_DETECTION)
print_problem_type_info("Object Detection", object_detection_props)
```

## Semantic Segmentation

Semantic segmentation performs pixel-level classification of images.


```python
# Semantic Segmentation
segmentation_props = PROBLEM_TYPES_REG.get(SEMANTIC_SEGMENTATION)
print_problem_type_info("Semantic Segmentation", segmentation_props)
```

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

## Named Entity Recognition (NER)

NER identifies and classifies named entities (like person names, locations, organizations) in text.


```python
# Named Entity Recognition
ner_props = PROBLEM_TYPES_REG.get(NER)
print_problem_type_info("Named Entity Recognition", ner_props)
```

## Feature Extraction

Feature extraction converts raw data into meaningful feature vector.


```python
# Feature Extraction
feature_extraction_props = PROBLEM_TYPES_REG.get(FEATURE_EXTRACTION)
print_problem_type_info("Feature Extraction", feature_extraction_props)
```

## Few-shot Classification

Few-shot classification learns to classify from a small number of examples per class.


```python
# Few-shot Classification
few_shot_props = PROBLEM_TYPES_REG.get(FEW_SHOT_CLASSIFICATION)
print_problem_type_info("Few-shot Classification", few_shot_props)
```

## Other Examples
You may go to [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm) to explore other examples about AutoMM.

## Customization
To learn how to customize AutoMM, please refer to [Customize AutoMM](../advanced_topics/customization.ipynb).
