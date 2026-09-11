# Condensed: ```python

Summary: This tutorial covers implementing Named Entity Recognition (NER) using AutoGluon's MultiModalPredictor with specific JSON annotation format requirements (entity_group, start, end keys). It demonstrates training NER models with customizable backbones (e.g., ELECTRA), evaluating with seqeval metrics (precision, recall, F1), making predictions with probability outputs, and visualizing results. Key techniques include: setting `problem_type="ner"`, configuring hyperparameters for backbone models, continuous training from saved checkpoints, and entity-specific metric evaluation. Supports BIO tagging format and provides utilities for annotation visualization in Jupyter notebooks. Useful for building custom NER systems, fine-tuning pre-trained models, and extracting named entities from text with confidence scores.

*This is a condensed version that preserves essential implementation details and context.*

# Named Entity Recognition (NER) with AutoMM

## Installation & Data Format

```python
!pip install autogluon.multimodal
```

**Required JSON annotation format:**
```python
[
    {"entity_group": "PERSON", "start": 0, "end": 15},
    {"entity_group": "LOCATION", "start": 28, "end": 35}
]
```

- **entity_group**: Entity category
- **start**: Character-level beginning position
- **end**: Character-level ending position

⚠️ **Critical**: Use exact key names (entity_group, start, end) for AutoMM compatibility.

**Visualization:**
```python
from autogluon.multimodal.utils import visualize_ner

sentence = "Albert Einstein was born in Germany..."
annotation = [{"entity_group": "PERSON", "start": 0, "end": 15},
              {"entity_group": "LOCATION", "start": 28, "end": 35}]
visualize_ner(sentence, annotation)
```

**BIO Format Support**: Optional B-prefix (beginning) and I-prefix (inside) tagging supported. O-tags handled automatically.

## Training

```python
from autogluon.multimodal import MultiModalPredictor

label_col = "entity_annotations"
predictor = MultiModalPredictor(problem_type="ner", label=label_col, path="./automm_ner")
predictor.fit(
    train_data=train_data,
    hyperparameters={'model.ner_text.checkpoint_name':'google/electra-small-discriminator'},
    time_limit=300  # seconds
)
```

**Key Parameters:**
- `problem_type="ner"` (required)
- `time_limit`: Recommended 30-60 minutes for production
- `hyperparameters`: Customize backbone model

## Evaluation

Uses [seqeval](https://huggingface.co/spaces/evaluate-metric/seqeval) metrics:

```python
predictor.evaluate(test_data, metrics=['overall_recall', "overall_precision", "overall_f1", "actor"])
```

**Supported metrics**: overall_recall, overall_precision, overall_f1, overall_accuracy, or specific entity group names

## Prediction & Visualization

```python
sentence = "Game of Thrones is an American fantasy drama..."
predictions = predictor.predict({'text_snippet': [sentence]})
print('Predicted entities:', predictions[0])

# Visualize in Jupyter
visualize_ner(sentence, predictions[0])
```

**Prediction probabilities:**
```python
predictions = predictor.predict_proba({'text_snippet': [sentence]})
print(predictions[0][0]['probability'])
```

## Reloading & Continuous Training

```python
# Reload
new_predictor = MultiModalPredictor.load(model_path)

# Continue training
new_predictor.fit(train_data, time_limit=60, save_path=new_model_path)
test_score = new_predictor.evaluate(test_data, metrics=['overall_f1', 'ACTOR'])
```

**See also**: [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm) | [Customize AutoMM](../advanced_topics/customization.ipynb)