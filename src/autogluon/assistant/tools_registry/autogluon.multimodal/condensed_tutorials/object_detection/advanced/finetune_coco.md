# Condensed: ```python

Summary: This tutorial demonstrates object detection using AutoGluon MultiModal, focusing on fine-tuning YOLOX models on custom datasets in COCO format. It covers installation of required packages, dataset preparation, model configuration with two-stage learning rates, training with early stopping, evaluation using mAP metrics, and result visualization. Key functionalities include configuring GPU usage, batch size optimization, using quality presets for simplified workflows, and performance tuning options. The tutorial helps with implementing custom object detection systems, visualizing detection results, and optimizing model performance through hyperparameter adjustments.

*This is a condensed version that preserves essential implementation details and context.*

# Object Detection with AutoGluon MultiModal

## Installation

```python
# Install required packages
!pip install autogluon.multimodal
!pip install -U pip setuptools wheel
!sudo apt-get install -y ninja-build gcc g++

# Install MMCV and related libraries
!python3 -m mim install "mmcv==2.1.0"
# Alternative for Colab: pip install "mmcv==2.1.0" -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html
!python3 -m pip install "mmdet==3.2.0"
!python3 -m pip install "mmengine>=0.10.6"
```

## Setup and Data Preparation

```python
from autogluon.multimodal import MultiModalPredictor
import os
from autogluon.core.utils.loaders import load_zip

# Download and extract dataset
zip_file = "https://automl-mm-bench.s3.amazonaws.com/object_detection/dataset/pothole.zip"
download_dir = "./pothole"
load_zip.unzip(zip_file, unzip_dir=download_dir)

# Set paths for COCO format annotation files
data_dir = os.path.join(download_dir, "pothole")
train_path = os.path.join(data_dir, "Annotations", "usersplit_train_cocoformat.json")
val_path = os.path.join(data_dir, "Annotations", "usersplit_val_cocoformat.json")
test_path = os.path.join(data_dir, "Annotations", "usersplit_test_cocoformat.json")
```

## Model Configuration and Training

```python
# Select model and GPU configuration
checkpoint_name = "yolox_s"  # YOLOX-small model
num_gpus = 1

# Initialize predictor
predictor = MultiModalPredictor(
    hyperparameters={
        "model.mmdet_image.checkpoint_name": checkpoint_name,
        "env.num_gpus": num_gpus,
    },
    problem_type="object_detection",
    sample_data_path=train_path,  # Used to infer dataset categories
)

# Train the model
predictor.fit(
    train_path,
    tuning_data=val_path,
    hyperparameters={
        "optim.lr": 1e-4,  # Two-stage learning rate (detection head has 100x lr)
        "env.per_gpu_batch_size": 16,  # Adjust based on GPU memory
        "optim.max_epochs": 30,
        "optim.val_check_interval": 1.0,  # Validate once per epoch
        "optim.check_val_every_n_epoch": 3,  # Validate every 3 epochs
        "optim.patience": 3,  # Early stopping after 3 non-improving validations
    },
)
```

## Evaluation and Prediction

```python
# Evaluate model on test set
predictor.evaluate(test_path)  # Returns mAP (COCO standard) and mAP50 (VOC standard)

# Get predictions
pred = predictor.predict(test_path)
```

## Visualization

```python
!pip install opencv-python

from autogluon.multimodal.utils import visualize_detection
from PIL import Image
from IPython.display import display

# Visualize detection results
conf_threshold = 0.25  # Filter out low-confidence predictions
visualization_result_dir = "./"
visualized = visualize_detection(
    pred=pred[12:13],
    detection_classes=predictor.classes,
    conf_threshold=conf_threshold,
    visualization_result_dir=visualization_result_dir,
)
img = Image.fromarray(visualized[0][:, :, ::-1], 'RGB')
display(img)
```

## Using Presets (Recommended)

```python
# Alternative simplified approach using presets
predictor = MultiModalPredictor(
    problem_type="object_detection",
    sample_data_path=train_path,
    presets="medium_quality",  # Options: "medium_quality", "high_quality", "best_quality"
)
predictor.fit(train_path, tuning_data=val_path)
predictor.evaluate(test_path)
```

## Key Notes

- Two-stage learning rate with higher rates for head layers improves convergence speed and performance
- Adjust batch size based on available GPU memory
- For higher performance, consider using larger models or the "high_quality"/"best_quality" presets
- See "AutoMM Detection - High Performance Finetune on COCO Format Dataset" tutorial for advanced configurations