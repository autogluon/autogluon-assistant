# Condensed: AutoMM Detection - Object detection data formats

Summary: This tutorial demonstrates object detection implementation using AutoGluon's MultiModalPredictor, covering two key data formats: COCO JSON and DataFrame. It provides code for format conversion, model training configuration, and evaluation. The tutorial specifically helps with tasks like setting up object detection training pipelines, data format handling, and model evaluation. Key features include support for COCO-format annotations, DataFrame conversions, YOLOv3 model implementation, GPU utilization, learning rate configuration, and batch size optimization. The implementation details are particularly useful for understanding data structure requirements, model hyperparameter tuning, and essential dependencies setup for AutoGluon-based object detection systems.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details:

# AutoMM Detection Data Formats

## Supported Data Formats

### 1. COCO Format
Requires a JSON file with the following structure:
```python
{
    "categories": [
        {"id": 1, "name": "person", "supercategory": "none"},
        # ...
    ],
    "images": [
        {
            "file_name": "image1.jpg",
            "height": 427,
            "width": 640,
            "id": 1
        },
        # ...
    ],
    "annotations": [
        {
            'area': 33453,
            'bbox': [x, y, width, height],
            'category_id': 8,
            'image_id': 1617,
            'id': 1,
            'iscrowd': 0,
            'ignore': 0,
            'segmentation': []
        },
        # ...
    ],
    "type": "instances"
}
```

### 2. DataFrame Format
Requires DataFrame with 3 columns:
- `image`: path to image file
- `rois`: list of arrays `[x1, y1, x2, y2, class_label]`
- `label`: copy of `rois`

## Implementation

### Format Conversion
```python
# COCO to DataFrame
from autogluon.multimodal.utils.object_detection import from_coco
train_df = from_coco(train_path)

# DataFrame to COCO
from autogluon.multimodal.utils.object_detection import object_detection_df_to_coco
train_coco = object_detection_df_to_coco(train_df, save_path="output.json")
```

### Training Setup
```python
from autogluon.multimodal import MultiModalPredictor

predictor = MultiModalPredictor(
    hyperparameters={
        "model.mmdet_image.checkpoint_name": "yolov3_mobilenetv2_320_300e_coco",
        "env.num_gpus": -1,  # use all GPUs
    },
    problem_type="object_detection",
    sample_data_path=train_df,
    path=model_path,
)

# Training
predictor.fit(
    train_df,
    hyperparameters={
        "optimization.learning_rate": 2e-4,
        "optimization.max_epochs": 30,
        "env.per_gpu_batch_size": 32,
    },
)
```

### Evaluation
```python
test_df = from_coco(test_path)
predictor.evaluate(test_df)
```

## Important Notes
- When loading from saved COCO JSON, ensure correct root path for images
- Required dependencies: `mmcv` and `mmdet==3.1.0`
- Adjust batch size based on model size and available memory
- Learning rate uses two-stage training with detection head having 100x lr

## Prerequisites
```bash
pip install autogluon.multimodal
mim install mmcv
pip install "mmdet==3.1.0"
```