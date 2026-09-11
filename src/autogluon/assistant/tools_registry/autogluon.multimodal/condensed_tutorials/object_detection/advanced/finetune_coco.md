# Condensed: ```python

Summary: This tutorial covers AutoGluon MultiModalPredictor implementation for object detection tasks using COCO format datasets. It demonstrates installing dependencies (MMCV, MMDet, MMEngine), loading COCO-formatted annotations, configuring YOLOX models with customizable checkpoints, and fine-tuning with two-stage learning rates (100x on detection head for faster convergence). Key techniques include hyperparameter tuning (batch size, learning rate, validation intervals, early stopping), model evaluation using mAP metrics, generating predictions, and visualizing detection results with confidence thresholds. The tutorial supports both manual hyperparameter configuration and preset-based workflows ("medium_quality", "high_quality", "best_quality") for rapid prototyping. Useful for implementing end-to-end object detection pipelines, fine-tuning pretrained models on custom datasets, and optimizing training performance on small datasets.

*This is a condensed version that preserves essential implementation details and context.*

# Object Detection Tutorial - Condensed

## Installation

```python
!pip install autogluon.multimodal

# Update package tools and install build dependencies
!pip install -U pip setuptools wheel
!sudo apt-get install -y ninja-build gcc g++

# Install MMCV
!python3 -m mim install "mmcv==2.1.0"

# For Google Colab: Alternative MMCV installation if above fails
# pip install "mmcv==2.1.0" -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html

# Install MMDet and MMEngine (>=0.10.6 for PyTorch 2.5 compatibility)
!python3 -m pip install "mmdet==3.2.0"
!python3 -m pip install "mmengine>=0.10.6"
```

## Dataset Setup (COCO Format)

```python
from autogluon.multimodal import MultiModalPredictor
import os
from autogluon.core.utils.loaders import load_zip

# Download and prepare dataset
zip_file = "https://automl-mm-bench.s3.amazonaws.com/object_detection/dataset/pothole.zip"
download_dir = "./pothole"
load_zip.unzip(zip_file, unzip_dir=download_dir)

data_dir = os.path.join(download_dir, "pothole")
train_path = os.path.join(data_dir, "Annotations", "usersplit_train_cocoformat.json")
val_path = os.path.join(data_dir, "Annotations", "usersplit_val_cocoformat.json")
test_path = os.path.join(data_dir, "Annotations", "usersplit_test_cocoformat.json")
```

**Note**: Input is the JSON annotation file for each dataset split in COCO format.

## Model Configuration

```python
checkpoint_name = "yolox_s"  # YOLOX-small for fast training/inference
num_gpus = 1

# Create predictor with problem_type="object_detection"
predictor = MultiModalPredictor(
    hyperparameters={
        "model.mmdet_image.checkpoint_name": checkpoint_name,
        "env.num_gpus": num_gpus,
    },
    problem_type="object_detection",
    sample_data_path=train_path,  # Required to infer dataset categories
)
```

**Key Points**:
- Use larger models via `checkpoint_name` for better performance (slower speed)
- Adjust `lr` and `per_gpu_batch_size` for different models
- Alternative: Use presets (`"medium_quality"`, `"high_quality"`, `"best_quality"`)

## Training

```python
predictor.fit(
    train_path,
    tuning_data=val_path,
    hyperparameters={
        "optim.lr": 1e-4,  # Two-stage LR: head has 100x learning rate
        "env.per_gpu_batch_size": 16,  # Adjust based on GPU memory
        "optim.max_epochs": 30,
        "optim.val_check_interval": 1.0,  # Validate once per epoch
        "optim.check_val_every_n_epoch": 3,  # Validate every 3 epochs
        "optim.patience": 3,  # Early stopping after 3 non-improving validations
    },
)
```

**Important**: Two-stage learning rate (100x on head) enables faster convergence and better performance, especially on small datasets.

## Evaluation & Prediction

```python
# Evaluate on test set
predictor.evaluate(test_path)  # Returns mAP (COCO) and mAP50 (VOC)

# Generate predictions
pred = predictor.predict(test_path)
```

## Visualization

```python
!pip install opencv-python

from autogluon.multimodal.utils import visualize_detection
from PIL import Image
from IPython.display import display

conf_threshold = 0.25  # Filter low-confidence boxes
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

## Recommended: Using Presets

```python
predictor = MultiModalPredictor(
    problem_type="object_detection",
    sample_data_path=train_path,
    presets="medium_quality",  # Saves customization time
)
predictor.fit(train_path, tuning_data=val_path)
predictor.evaluate(test_path)
```

## Performance Notes

- Fast finetune achieves good mAP in hundreds of seconds
- For higher performance: VFNet with 5-hour training reached `mAP=0.450, mAP50=0.718`
- See [High Performance Finetune guide](../finetune/detection_high_performance_finetune_coco.ipynb) for advanced optimization