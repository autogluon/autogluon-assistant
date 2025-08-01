# Condensed: ```python

Summary: This tutorial demonstrates implementing semantic segmentation using Meta's Segment Anything Model (SAM) with AutoGluon. It covers zero-shot inference and fine-tuning SAM for domain-specific tasks using LoRA (Low-Rank Adaptation). Key functionalities include data preparation for segmentation tasks, model initialization with pre-trained SAM checkpoints, efficient fine-tuning techniques, evaluation using IoU metrics, and visualization of segmentation masks. The tutorial helps with implementing image segmentation tasks that require minimal labeled data by leveraging transfer learning from foundation models, particularly useful for specialized domains like disease detection in plant leaves.

*This is a condensed version that preserves essential implementation details and context.*

# Semantic Segmentation with SAM (Segment Anything Model)

## Setup and Data Preparation

```python
!pip install autogluon.multimodal

# Download and extract dataset
download_dir = './ag_automm_tutorial'
zip_file = 'https://automl-mm-bench.s3.amazonaws.com/semantic_segmentation/leaf_disease_segmentation.zip'
from autogluon.core.utils.loaders import load_zip
load_zip.unzip(zip_file, unzip_dir=download_dir)

# Load data
import pandas as pd
import os
dataset_path = os.path.join(download_dir, 'leaf_disease_segmentation')
train_data = pd.read_csv(f'{dataset_path}/train.csv', index_col=0)
val_data = pd.read_csv(f'{dataset_path}/val.csv', index_col=0)
test_data = pd.read_csv(f'{dataset_path}/test.csv', index_col=0)
image_col = 'image'
label_col = 'label'

# Expand relative paths to absolute paths
def path_expander(path, base_folder):
    path_l = path.split(';')
    return ';'.join([os.path.abspath(os.path.join(base_folder, path)) for path in path_l])

for per_col in [image_col, label_col]:
    train_data[per_col] = train_data[per_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
    val_data[per_col] = val_data[per_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
    test_data[per_col] = test_data[per_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
```

## Zero-Shot Evaluation

```python
from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.utils import SemanticSegmentationVisualizer

# Initialize visualizer
visualizer = SemanticSegmentationVisualizer()

# Initialize zero-shot predictor with SAM base model
predictor_zero_shot = MultiModalPredictor(
    problem_type="semantic_segmentation", 
    label=label_col,
    hyperparameters={
        "model.sam.checkpoint_name": "facebook/sam-vit-base",
    },
    num_classes=1  # foreground-background segmentation
)

# Perform inference
pred_zero_shot = predictor_zero_shot.predict({'image': [test_data.iloc[0]['image']]})

# Evaluate on test data
scores = predictor_zero_shot.evaluate(test_data, metrics=["iou"])
print(scores)
```

**Note:** SAM without prompts outputs a rough leaf mask instead of disease masks due to lack of context about the domain task.

## Fine-tuning SAM

```python
import uuid
save_path = f"./tmp/{uuid.uuid4().hex}-automm_semantic_seg"

# Initialize predictor for fine-tuning
predictor = MultiModalPredictor(
    problem_type="semantic_segmentation", 
    label="label",
    hyperparameters={
        "model.sam.checkpoint_name": "facebook/sam-vit-base",
    },
    path=save_path,
)

# Fine-tune using LoRA (Low-Rank Adaptation)
predictor.fit(
    train_data=train_data,
    tuning_data=val_data,
    time_limit=180,  # seconds
)

# Evaluate fine-tuned model
scores = predictor.evaluate(test_data, metrics=["iou"])
print(scores)

# Visualize prediction
pred = predictor.predict({'image': [test_data.iloc[0]['image']]})
visualizer.plot_mask(pred)
```

## Save and Load

```python
# The model is automatically saved during fit()
# To load:
loaded_predictor = MultiModalPredictor.load(save_path)
scores = loaded_predictor.evaluate(test_data, metrics=["iou"])
print(scores)
```

**Warning:** `MultiModalPredictor.load()` uses the `pickle` module, which can execute arbitrary code during unpickling. Only load data from trusted sources.

## Implementation Details

- Uses LoRA for efficient fine-tuning of the large SAM model
- Fine-tuning significantly improves performance for domain-specific segmentation tasks
- The base SAM model is used as default without hyperparameter customization
- For more examples, see [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm)
- For customization options, refer to [Customize AutoMM](../advanced_topics/customization.ipynb)