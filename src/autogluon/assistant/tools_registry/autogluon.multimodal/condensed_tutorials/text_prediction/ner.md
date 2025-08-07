# Condensed: ```python

Summary: "Summary: 
Summary: 
Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: This tutorial demonstrates implementing Named Entity Recognition (NER) with AutoGluon MultiModal. It covers the JSON annotation format required for NER tasks (with entity_group, start, and end positions), model training using MultiModalPredictor, evaluation with metrics like F1 score, and prediction visualization. The tutorial shows how to train NER models with pre-trained checkpoints like ELECTRA, make predictions on new text, extract prediction probabilities, and reload/continue training models. It provides practical code examples for implementing complete NER workflows with AutoGluon's simplified API.

*This is a condensed version that preserves essential implementation details and context.*

# Named Entity Recognition with AutoGluon MultiModal

## Installation and Setup

```python
!pip install autogluon.multimodal
```

## Data Format

NER annotations require JSON format with specific keys:
```python
json.dumps([
    {"entity_group": "PERSON", "start": 0, "end": 15},
    {"entity_group": "LOCATION", "start": 28, "end": 35}
])
```

Key requirements:
- `entity_group`: category of the entity
- `start`: character position where entity begins
- `end`: character position where entity ends

## Visualizing Annotations

```python
from autogluon.multimodal.utils import visualize_ner

sentence = "Albert Einstein was born in Germany and is widely acknowledged to be one of the greatest physicists."
annotation = [{"entity_group": "PERSON", "start": 0, "end": 15},
              {"entity_group": "LOCATION", "start": 28, "end": 35}]

visualize_ner(sentence, annotation)
```

Note: BIO (Beginning-Inside-Outside) format is supported but not required.

## Training

```python
from autogluon.multimodal import MultiModalPredictor
import uuid

label_col = "entity_annotations"
model_path = f"./tmp/{uuid.uuid4().hex}-automm_ner"
predictor = MultiModalPredictor(problem_type="ner", label=label_col, path=model_path)
predictor.fit(
    train_data=train_data,
    hyperparameters={'model.ner_text.checkpoint_name':'google/electra-small-discriminator'},
    time_limit=300, # seconds
)
```

Important: For production use, set longer `time_limit` (30-60 minutes recommended).

## Evaluation

```python
predictor.evaluate(test_data, metrics=['overall_recall', "overall_precision", "overall_f1", "actor"])
```

Supported metrics:
- `overall_recall`, `overall_precision`, `overall_f1`, `overall_accuracy`
- Entity-specific metrics (e.g., "actor")

## Prediction and Visualization

```python
sentence = "Game of Thrones is an American fantasy drama television series created by David Benioff"
predictions = predictor.predict({'text_snippet': [sentence]})
visualize_ner(sentence, predictions[0])
```

## Prediction Probabilities

```python
predictions = predictor.predict_proba({'text_snippet': [sentence]})
print(predictions[0][0]['probability'])
```

## Model Reloading and Continuous Training

```python
new_predictor = MultiModalPredictor.load(model_path)
new_model_path = f"./tmp/{uuid.uuid4().hex}-automm_ner_continue_train"
new_predictor.fit(train_data, time_limit=60, save_path=new_model_path)
test_score = new_predictor.evaluate(test_data, metrics=['overall_f1', 'ACTOR'])
```