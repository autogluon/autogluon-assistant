# Condensed: AutoMM for Named Entity Recognition - Quick Start

Summary: This tutorial demonstrates implementing Named Entity Recognition (NER) using AutoGluon's MultiModalPredictor. It covers essential techniques for data preparation with specific JSON annotation formats, model training configuration using BERT-based or ELECTRA models, and prediction workflows. The tutorial helps with tasks like setting up NER training pipelines, model evaluation using seqeval metrics, and making predictions with probability scores. Key features include flexible model selection, customizable training parameters, support for continuous training, built-in visualization tools, and comprehensive evaluation metrics including entity-specific measurements. The implementation focuses on practical aspects like proper data formatting, model persistence, and best practices for production deployment.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details:

# AutoMM for Named Entity Recognition - Quick Start

## Data Preparation
- Required format: DataFrame with text column and annotation column
- Annotation format must be JSON with specific structure:
```python
[{
    "entity_group": "CATEGORY",
    "start": char_start_position,
    "end": char_end_position
}]
```

Example:
```python
annotation = [
    {"entity_group": "PERSON", "start": 0, "end": 15},
    {"entity_group": "LOCATION", "start": 28, "end": 35}
]
```

## Implementation

1. Install and import:
```python
!pip install autogluon.multimodal
from autogluon.multimodal import MultiModalPredictor
```

2. Training setup:
```python
predictor = MultiModalPredictor(
    problem_type="ner", 
    label="entity_annotations",
    path="model_path"
)

# Training
predictor.fit(
    train_data=train_data,
    hyperparameters={
        'model.ner_text.checkpoint_name':'google/electra-small-discriminator'
    },
    time_limit=300  # in seconds
)
```

3. Evaluation:
```python
metrics = predictor.evaluate(
    test_data,  
    metrics=['overall_recall', 'overall_precision', 'overall_f1']
)
```

4. Prediction:
```python
# Basic prediction
predictions = predictor.predict({'text_snippet': [text]})

# Prediction with probabilities
prob_predictions = predictor.predict_proba({'text_snippet': [text]})
```

## Key Configurations & Best Practices

1. Model Selection:
- Default: BERT-based models
- Lightweight option: 'google/electra-small-discriminator'

2. Training Parameters:
- Recommended time_limit: 30-60 minutes for production use
- Can specify custom backbone models via hyperparameters

3. Evaluation Metrics:
- Uses seqeval metrics
- Available metrics: overall_recall, overall_precision, overall_f1, overall_accuracy
- Entity-specific metrics available using entity group names

## Important Notes

1. Data Format:
- Must use exact keys: "entity_group", "start", "end"
- BIO format supported but not required

2. Model Management:
- Models auto-save during training
- Can reload models using:
```python
new_predictor = MultiModalPredictor.load(model_path)
```
- Supports continuous training on new data

3. Visualization:
```python
from autogluon.multimodal.utils import visualize_ner
visualize_ner(text, annotations)  # Works in Jupyter notebooks
```