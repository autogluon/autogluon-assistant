# Condensed: AutoMM Detection - Object detection data formats

Summary: Summary: 

Summary: Summary: This tutorial covers object detection data formats for AutoMM's AutoMM Detection, focusing on COCO and DataFrame formats for object detection. Supports both COCO and DataFrame formats for training and evaluation. Supports conversion between formats with utilities for object detection data formats. Supports training with AutoMM's MultiModalPrediagramming formats for object detection. Supports training with AutoMM's MultiModalPredictor. Supports object detection with AutoMM's MultiModalPredictor. Supports object detection with COCO and DataFrame formats for training and evaluation. Supports object detection with AutoMM's MultiModalPredictor. Supports object detection with AutoMM's MultiModalPredictor. Supports object detection with AutoMM's MultiModalPredictor. Supports object detection with AutoMM's MultiModalPredictor.

Summary: This tutorial explains AutoMM's object detection data format handling, covering: 1) Implementation of COCO and DataFrame formats for object detection tasks, 2) Utility functions for converting between these formats (from_coco and object_detection_df_to_coco), 3) Training and evaluating object detection models using MultiModalPredictor with customizable hyperparameters. The tutorial demonstrates how to properly structure bounding box data, configure model checkpoints, and set training parameters for computer vision object detection tasks.

*This is a condensed version that preserves essential implementation details and context.*

# AutoMM Detection - Object Detection Data Formats

## Supported Data Formats

### 1. COCO Format
Requires a `.json` file with the following structure:

```python
data = {
    "categories": [
        {"supercategory": "none", "id": 1, "name": "person"},
        {"supercategory": "none", "id": 2, "name": "bicycle"},
        # ...
    ],
    "images": [
        {
            "file_name": "<imagename0>.<ext>",
            "height": 427,
            "width": 640,
            "id": 1
        },
        # ...
    ],
    "annotations": [
        {
            'area': 33453,
            'iscrowd': 0,
            'bbox': [181, 133, 177, 189],  # [x, y, width, height]
            'category_id': 8,
            'ignore': 0,
            'segmentation': [],
            'image_id': 1617,
            'id': 1
        },
        # ...
    ],
    "type": "instances"
}
```

### 2. DataFrame Format
Requires a `pd.DataFrame` with 3 columns:
- `image`: path to image file
- `rois`: list of arrays with format `[x1, y1, x2, y2, class_label]`
- `label`: copy of `rois` column

## Format Conversion Utilities

### COCO to DataFrame
```python
from autogluon.multimodal.utils.object_detection import from_coco
train_df = from_coco(train_path)
```

### DataFrame to COCO
```python
from autogluon.multimodal.utils.object_detection import object_detection_df_to_coco
train_coco = object_detection_df_to_coco(train_df, save_path="./df_converted_to_coco.json")
```

When loading from saved COCO file, specify the correct root path:
```python
train_df_from_saved_coco = from_coco("./df_converted_to_coco.json", root="./")
```

## Training with DataFrame Format

```python
from autogluon.multimodal import MultiModalPredictor

# Required dependencies
# mim install mmcv
# pip install "mmdet==3.1.0"

checkpoint_name = "yolov3_mobilenetv2_320_300e_coco"
num_gpus = -1  # use all GPUs

predictor_df = MultiModalPredictor(
    hyperparameters={
        "model.mmdet_image.checkpoint_name": checkpoint_name,
        "env.num_gpus": num_gpus,
    },
    problem_type="object_detection",
    sample_data_path=train_df,  # needed to determine num_classes
    path=model_path,
)

predictor_df.fit(
    train_df,
    hyperparameters={
        "optim.lr": 2e-4,  # detection head has 100x lr
        "optim.max_epochs": 30,
        "env.per_gpu_batch_size": 32,  # decrease for larger models
    },
)
```

## Evaluation with DataFrame Format
```python
test_df = from_coco(test_path)
predictor_df.evaluate(test_df)
```

For more examples, see [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm).

For customization options, refer to [Customize AutoMM](../../advanced_topics/customization.ipynb).