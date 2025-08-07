# Condensed: ```python

Summary: This tutorial demonstrates implementing multimodal named entity recognition using AutoGluon, showing how to process text and image data together for NER tasks. It covers dataset preparation with image path handling, model training with the MultiModalPredictor (specifying "ner" problem type and "text_ner" column type), evaluation using precision/recall/F1 metrics, entity prediction with detailed output parsing, and model persistence. Key features include automatic modality detection, late fusion of multimodal data, and support for continuous training. This implementation helps with building NER systems that leverage both textual and visual information to identify and classify named entities.

*This is a condensed version that preserves essential implementation details and context.*

# Multimodal Named Entity Recognition with AutoGluon

## Setup

```python
!pip install autogluon.multimodal

import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
```

## Dataset Preparation

```python
# Download Twitter dataset
download_dir = './ag_automm_tutorial_ner'
zip_file = 'https://automl-mm-bench.s3.amazonaws.com/ner/multimodal_ner.zip'
from autogluon.core.utils.loaders import load_zip
load_zip.unzip(zip_file, unzip_dir=download_dir)

# Load data
dataset_path = download_dir + '/multimodal_ner'
train_data = pd.read_csv(f'{dataset_path}/twitter17_train.csv')
test_data = pd.read_csv(f'{dataset_path}/twitter17_test.csv')
label_col = 'entity_annotations'

# Process image paths
image_col = 'image'
train_data[image_col] = train_data[image_col].apply(lambda ele: ele.split(';')[0])  # Use first image only
test_data[image_col] = test_data[image_col].apply(lambda ele: ele.split(';')[0])

def path_expander(path, base_folder):
    path_l = path.split(';')
    p = ';'.join([os.path.abspath(base_folder+path) for path in path_l])
    return p

train_data[image_col] = train_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
test_data[image_col] = test_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
```

## Training

```python
from autogluon.multimodal import MultiModalPredictor
import uuid

model_path = f"./tmp/{uuid.uuid4().hex}-automm_multimodal_ner"
predictor = MultiModalPredictor(problem_type="ner", label=label_col, path=model_path)
predictor.fit(
    train_data=train_data,
    column_types={"text_snippet":"text_ner"},  # Important: specify text_ner column type
    time_limit=300,  # seconds
)
```

## Evaluation

```python
predictor.evaluate(test_data, metrics=['overall_recall', "overall_precision", "overall_f1"])
```

## Prediction

```python
prediction_input = test_data.drop(columns=label_col).head(1)
predictions = predictor.predict(prediction_input)
print('Tweet:', prediction_input.text_snippet[0])
print('Image path:', prediction_input.image[0])
print('Predicted entities:', predictions[0])

for entity in predictions[0]:
    print(f"Word '{prediction_input.text_snippet[0][entity['start']:entity['end']]}' belongs to group: {entity['entity_group']}")
```

## Model Reloading and Continuous Training

```python
new_predictor = MultiModalPredictor.load(model_path)
new_model_path = f"./tmp/{uuid.uuid4().hex}-automm_multimodal_ner_continue_train"
new_predictor.fit(train_data, time_limit=60, save_path=new_model_path)
test_score = new_predictor.evaluate(test_data, metrics=['overall_f1'])
print(test_score)
```

## Key Implementation Details

- **Problem Type**: Set to "ner" for named entity recognition
- **Column Types**: Specify "text_ner" for the text column containing entities
- **Multimodal Processing**: AutoMM automatically detects data modalities and selects appropriate models
- **Late Fusion**: When multiple backbones are available, AutoMM appends a late-fusion model on top

For customization options, refer to the "Customize AutoMM" documentation.