# Condensed: ```python

Summary: This tutorial demonstrates fine-tuning Meta's Segment Anything Model (SAM) for semantic segmentation using AutoGluon's MultiModalPredictor. It covers essential techniques including: data preparation with path expansion for image/mask pairs in pandas DataFrames, zero-shot inference with pretrained SAM models, and efficient fine-tuning using LoRA. Key implementations include configuring the predictor with `problem_type="semantic_segmentation"`, specifying SAM checkpoints (facebook/sam-vit-base), training with time limits, and evaluating with IoU metrics. The tutorial helps with tasks like custom segmentation model development, transfer learning workflows, and model persistence (save/load). Critical features include automatic LoRA-based fine-tuning, support for validation data during training, and direct prediction/evaluation methods for segmentation masks.

*This is a condensed version that preserves essential implementation details and context.*

# Fine-tuning SAM for Semantic Segmentation with AutoGluon

## Installation & Setup

```python
!pip install autogluon.multimodal
```

## Data Preparation

Download and load the dataset:

```python
from autogluon.core.utils.loaders import load_zip
import pandas as pd
import os

download_dir = './ag_automm_tutorial'
zip_file = 'https://automl-mm-bench.s3.amazonaws.com/semantic_segmentation/leaf_disease_segmentation.zip'
load_zip.unzip(zip_file, unzip_dir=download_dir)

dataset_path = os.path.join(download_dir, 'leaf_disease_segmentation')
train_data = pd.read_csv(f'{dataset_path}/train.csv', index_col=0)
val_data = pd.read_csv(f'{dataset_path}/val.csv', index_col=0)
test_data = pd.read_csv(f'{dataset_path}/test.csv', index_col=0)
```

**Critical**: Expand relative paths to absolute paths for correct data loading:

```python
def path_expander(path, base_folder):
    path_l = path.split(';')
    return ';'.join([os.path.abspath(os.path.join(base_folder, path)) for path in path_l])

image_col = 'image'
label_col = 'label'

for per_col in [image_col, label_col]:
    train_data[per_col] = train_data[per_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
    val_data[per_col] = val_data[per_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
    test_data[per_col] = test_data[per_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
```

## Zero-Shot Evaluation

Initialize predictor for zero-shot inference:

```python
from autogluon.multimodal import MultiModalPredictor

predictor_zero_shot = MultiModalPredictor(
    problem_type="semantic_segmentation", 
    label=label_col,
    hyperparameters={
        "model.sam.checkpoint_name": "facebook/sam-vit-base",
    },
    num_classes=1,  # foreground-background segmentation
)

# Direct inference without training
pred_zero_shot = predictor_zero_shot.predict({'image': [test_data.iloc[0]['image']]})
scores = predictor_zero_shot.evaluate(test_data, metrics=["iou"])
```

**Note**: Zero-shot SAM outputs rough masks without domain context. Performance is relatively low without fine-tuning.

## Fine-tuning SAM

Initialize and train predictor:

```python
import uuid

save_path = f"./tmp/{uuid.uuid4().hex}-automm_semantic_seg"
predictor = MultiModalPredictor(
    problem_type="semantic_segmentation", 
    label="label",
    hyperparameters={
        "model.sam.checkpoint_name": "facebook/sam-vit-base",
    },
    path=save_path,
)

predictor.fit(
    train_data=train_data,
    tuning_data=val_data,
    time_limit=180,  # seconds
)
```

**Key Implementation Detail**: Uses [LoRA](https://arxiv.org/abs/2106.09685) for efficient fine-tuning under the hood. Default model is SAM-huge without customization.

Evaluate fine-tuned model:

```python
scores = predictor.evaluate(test_data, metrics=["iou"])
pred = predictor.predict({'image': [test_data.iloc[0]['image']]})
```

## Save and Load

Model auto-saves after `fit()`:

```python
loaded_predictor = MultiModalPredictor.load(save_path)
scores = loaded_predictor.evaluate(test_data, metrics=["iou"])
```

**⚠️ Security Warning**: `MultiModalPredictor.load()` uses `pickle` module, which is insecure. Malicious pickle data can execute arbitrary code during unpickling. **Only load data from trusted sources.**

## Additional Resources

- [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm)
- [Customize AutoMM](../advanced_topics/customization.ipynb)