# Condensed: ```python

Summary: This tutorial demonstrates implementing multimodal Named Entity Recognition (NER) using AutoGluon's MultiModalPredictor on text-image datasets. It covers critical techniques including: dataset preparation with image path expansion, configuring `problem_type="ner"` and `column_types={"text_snippet":"text_ner"}` for proper entity extraction, training with automatic modality detection and late-fusion, evaluation with NER-specific metrics (recall, precision, F1), prediction with entity span extraction, and model persistence/reloading for continuous training. Helps with tasks involving entity recognition from combined text and image data, multimodal data preprocessing, AutoMM configuration for NER problems, and model lifecycle management (save/load/retrain).

*This is a condensed version that preserves essential implementation details and context.*

# Multimodal Named Entity Recognition (NER) with AutoGluon

## Installation & Setup
```python
!pip install autogluon.multimodal
import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
```

## Dataset Preparation

**Twitter Dataset**: Tweets from 2016-2017, each with one sentence and one image.

```python
download_dir = './ag_automm_tutorial_ner'
zip_file = 'https://automl-mm-bench.s3.amazonaws.com/ner/multimodal_ner.zip'
from autogluon.core.utils.loaders import load_zip
load_zip.unzip(zip_file, unzip_dir=download_dir)

dataset_path = download_dir + '/multimodal_ner'
train_data = pd.read_csv(f'{dataset_path}/twitter17_train.csv')
test_data = pd.read_csv(f'{dataset_path}/twitter17_test.csv')
label_col = 'entity_annotations'
```

**Critical**: Expand image paths before training:

```python
image_col = 'image'
train_data[image_col] = train_data[image_col].apply(lambda ele: ele.split(';')[0])
test_data[image_col] = test_data[image_col].apply(lambda ele: ele.split(';')[0])

def path_expander(path, base_folder):
    path_l = path.split(';')
    return ';'.join([os.path.abspath(base_folder+path) for path in path_l])

train_data[image_col] = train_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
test_data[image_col] = test_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
```

## Training

**Key Configurations**:
- `problem_type="ner"` - Required for NER tasks
- `column_types={"text_snippet":"text_ner"}` - **Critical**: Specifies which text column contains entities (required when multiple text columns exist)
- AutoMM auto-detects modalities and selects appropriate models with late-fusion for multiple backbones

```python
from autogluon.multimodal import MultiModalPredictor
import uuid

label_col = "entity_annotations"
model_path = f"./tmp/{uuid.uuid4().hex}-automm_multimodal_ner"
predictor = MultiModalPredictor(problem_type="ner", label=label_col, path=model_path)
predictor.fit(
    train_data=train_data,
    column_types={"text_snippet":"text_ner"},
    time_limit=300,  # seconds
)
```

## Evaluation & Prediction

```python
# Evaluation
predictor.evaluate(test_data, metrics=['overall_recall', "overall_precision", "overall_f1"])

# Prediction
prediction_input = test_data.drop(columns=label_col).head(1)
predictions = predictor.predict(prediction_input)

# Parse results
for entity in predictions[0]:
    word = prediction_input.text_snippet[0][entity['start']:entity['end']]
    print(f"Word '{word}' belongs to group: {entity['entity_group']}")
```

## Model Reloading & Continuous Training

```python
# Reload saved model
new_predictor = MultiModalPredictor.load(model_path)

# Continue training
new_model_path = f"./tmp/{uuid.uuid4().hex}-automm_multimodal_ner_continue_train"
new_predictor.fit(train_data, time_limit=60, save_path=new_model_path)
test_score = new_predictor.evaluate(test_data, metrics=['overall_f1'])
```

## Additional Resources
- [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm)
- [Customize AutoMM](../advanced_topics/customization.ipynb)