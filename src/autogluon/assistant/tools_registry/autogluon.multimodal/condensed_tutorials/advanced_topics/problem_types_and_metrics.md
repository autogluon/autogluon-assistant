# Condensed: ```python

Summary: This tutorial demonstrates how to query and inspect AutoGluon MultiModal's problem type registry using `PROBLEM_TYPES_REG` to discover supported ML tasks. It provides implementation patterns for accessing problem type properties (supported modalities, evaluation metrics, zero-shot capabilities) through constants like `BINARY`, `MULTICLASS`, `REGRESSION`, `OBJECT_DETECTION`, `SEMANTIC_SEGMENTATION`, `TEXT_SIMILARITY`, `IMAGE_SIMILARITY`, `IMAGE_TEXT_SIMILARITY`, `NER`, `FEATURE_EXTRACTION`, and `FEW_SHOT_CLASSIFICATION`. The tutorial helps with tasks like programmatically determining available problem types, checking supported input modalities for each task, identifying default evaluation metrics, and verifying zero-shot/training support. Key features include a reusable helper function for formatted problem type inspection and systematic exploration of AutoGluon's multimodal capabilities across vision, text, and cross-modal tasks.

*This is a condensed version that preserves essential implementation details and context.*

# AutoGluon MultiModal Problem Types - Condensed Guide

## Installation & Setup

```python
!pip install autogluon.multimodal
import warnings
warnings.filterwarnings('ignore')
```

## Helper Function for Problem Type Info

```python
from autogluon.multimodal.constants import *
from autogluon.multimodal.problem_types import PROBLEM_TYPES_REG

def print_problem_type_info(name: str, props):
    """Helper function to print problem type information"""
    print(f"\n=== {name} ===")
    print("\nSupported Input Modalities:")
    for modality in sorted(list(props.supported_modality_type)):
        print(f"- {modality}")
    if hasattr(props, 'supported_evaluation_metrics') and props.supported_evaluation_metrics:
        print("\nEvaluation Metrics:")
        for metric in sorted(list(props.supported_evaluation_metrics)):
            print(f"- {metric}{' (default)' if metric == props.fallback_evaluation_metric else ''}")
    if hasattr(props, 'support_zero_shot'):
        print("\nSpecial Capabilities:")
        print(f"- Zero-shot: {'Supported' if props.support_zero_shot else 'Not supported'}")
        print(f"- Training: {'Supported' if props.support_fit else 'Not supported'}")
```

## Supported Problem Types

### Classification (Binary & Multiclass)

```python
binary_props = PROBLEM_TYPES_REG.get(BINARY)
multiclass_props = PROBLEM_TYPES_REG.get(MULTICLASS)
print_problem_type_info("Binary Classification", binary_props)
print_problem_type_info("Multiclass Classification", multiclass_props)
```

### Regression

```python
regression_props = PROBLEM_TYPES_REG.get(REGRESSION)
print_problem_type_info("Regression", regression_props)
```

### Object Detection

```python
object_detection_props = PROBLEM_TYPES_REG.get(OBJECT_DETECTION)
print_problem_type_info("Object Detection", object_detection_props)
```

### Semantic Segmentation

```python
segmentation_props = PROBLEM_TYPES_REG.get(SEMANTIC_SEGMENTATION)
print_problem_type_info("Semantic Segmentation", segmentation_props)
```

### Similarity Matching

Supports text-to-text, image-to-image, and image-to-text similarity. See [Matching Tutorials](../semantic_matching/index.md).

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
    print(f"Zero-shot: {'Supported' if props.support_zero_shot else 'Not supported'}")
```

### Named Entity Recognition (NER)

```python
ner_props = PROBLEM_TYPES_REG.get(NER)
print_problem_type_info("Named Entity Recognition", ner_props)
```

### Feature Extraction

```python
feature_extraction_props = PROBLEM_TYPES_REG.get(FEATURE_EXTRACTION)
print_problem_type_info("Feature Extraction", feature_extraction_props)
```

### Few-shot Classification

```python
few_shot_props = PROBLEM_TYPES_REG.get(FEW_SHOT_CLASSIFICATION)
print_problem_type_info("Few-shot Classification", few_shot_props)
```

## Additional Resources

- **More Examples**: [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm)
- **Customization**: [Customize AutoMM](../advanced_topics/customization.ipynb)