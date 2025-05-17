# Condensed: AutoMM for Entity Extraction with Text and Image - Quick Start

Summary: This tutorial demonstrates implementing multimodal entity extraction using AutoGluon's MultiModalPredictor, specifically focusing on processing combined text and image data. It covers essential techniques for dataset preparation (including image path handling), model configuration for NER tasks, and automated modality fusion. Key functionalities include proper column type specification for NER, model training with time limits, evaluation using standard metrics (recall, precision, F1), prediction capabilities, and model persistence. The tutorial helps with tasks involving entity extraction from multimodal sources, model saving/loading, and continuous training workflows. It emphasizes best practices for NER configuration and handling multiple text columns while showcasing AutoMM's automatic modality detection and fusion features.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details:

# AutoMM for Multimodal Entity Extraction - Quick Start

## Key Setup
```python
!pip install autogluon.multimodal
from autogluon.multimodal import MultiModalPredictor
import pandas as pd
```

## Dataset Preparation
1. Dataset contains tweets with text and images
2. Key data preprocessing:

```python
# Expand image paths
def path_expander(path, base_folder):
    path_l = path.split(';')
    p = ';'.join([os.path.abspath(base_folder+path) for path in path_l])
    return p

# Apply to image column
train_data[image_col] = train_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
```

## Model Training

### Critical Configurations
```python
predictor = MultiModalPredictor(
    problem_type="ner",  # Specify NER task
    label="entity_annotations",
    path=model_path
)

# Training with essential parameters
predictor.fit(
    train_data=train_data,
    column_types={"text_snippet": "text_ner"},  # Important: Specify text_ner column
    time_limit=300  # Training time in seconds
)
```

### Key Implementation Notes:
- AutoMM automatically:
  - Detects data modalities
  - Selects models from multimodal pools
  - Implements late-fusion for multiple backbones

## Evaluation & Prediction
```python
# Evaluate
metrics = predictor.evaluate(test_data, metrics=['overall_recall', "overall_precision", "overall_f1"])

# Predict
predictions = predictor.predict(prediction_input)
```

## Model Persistence & Continuous Training
```python
# Load saved model
new_predictor = MultiModalPredictor.load(model_path)

# Continue training
new_predictor.fit(
    train_data, 
    time_limit=60,
    save_path=new_model_path
)
```

## Best Practices
1. Always specify `problem_type="ner"` for entity extraction
2. Use `column_types={"text_snippet":"text_ner"}` when multiple text columns exist
3. Ensure image paths are properly expanded before training
4. Consider continuous training for improving model performance

## Important Notes
- Supports multimodal data (text + images)
- Automatically handles modality fusion
- Saves models for later use
- Allows continuous training on new data