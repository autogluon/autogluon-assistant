# Condensed: AutoMM Detection - Object detection data formats

Summary: This tutorial covers AutoGluon's AutoMM Detection data format handling for object detection tasks. It teaches two data formats: COCO JSON format (with categories, images, and annotations including bbox coordinates in [x,y,width,height]) and pandas DataFrame format (with image paths, rois as [x1,y1,x2,y2,class_label], and label columns). Key implementations include bidirectional format conversion using `from_coco()` and `object_detection_df_to_coco()` utilities, training object detection models with MultiModalPredictor using YOLOv3/MobileNetV2 checkpoints, and model evaluation. Critical for tasks involving COCO dataset preparation, format conversion between COCO and DataFrame, configuring detection model hyperparameters (learning rates, batch sizes, GPU settings), and building end-to-end object detection pipelines with AutoGluon.

*This is a condensed version that preserves essential implementation details and context.*

# AutoMM Detection - Object Detection Data Formats

AutoMM Detection supports two major data formats: COCO format and DataFrame format.

## COCO Format

Requires a `.json` file with the following structure:

```python
data = {
    "categories": [
        {"supercategory": "none", "id": 1, "name": "person"},
        {"supercategory": "none", "id": 2, "name": "bicycle"},
        # ...
    ],
    "images": [
        {"file_name": "<imagename0>.<ext>", "height": 427, "width": 640, "id": 1},
        # ...
    ],
    "annotations": [
        {
            'area': 33453,
            'iscrowd': 0,  # 0 for single object per box
            'bbox': [181, 133, 177, 189],  # [x, y, width, height]
            'category_id': 8,  # use category "id", not "name"
            'ignore': 0,  # set to 1 to ignore annotation
            'segmentation': [],  # empty for detection
            'image_id': 1617,
            'id': 1
        },
        # ...
    ],
    "type": "instances"
}
```

## DataFrame Format

Requires 3 columns:
- `image`: path to image file
- `rois`: list of arrays `[x1, y1, x2, y2, class_label]`
- `label`: copy of `rois` column

## Format Conversion

**COCO to DataFrame:**
```python
from autogluon.multimodal.utils.object_detection import from_coco
train_df = from_coco(train_path)
```

**DataFrame to COCO:**
```python
from autogluon.multimodal.utils.object_detection import object_detection_df_to_coco
train_coco = object_detection_df_to_coco(train_df, save_path="./output.json")
```

**⚠️ Important:** When loading from saved `.json`, specify `root` parameter so `<root>/<file_name>` forms valid image paths:
```python
train_df = from_coco("./output.json", root="./")
```

## Training with DataFrame

```python
from autogluon.multimodal import MultiModalPredictor

predictor = MultiModalPredictor(
    hyperparameters={
        "model.mmdet_image.checkpoint_name": "yolov3_mobilenetv2_320_300e_coco",
        "env.num_gpus": -1,  # use all GPUs
    },
    problem_type="object_detection",
    sample_data_path=train_df,  # required for num_classes
    path=model_path,
)

predictor.fit(
    train_df,
    hyperparameters={
        "optim.lr": 2e-4,
        "optim.max_epochs": 30,
        "env.per_gpu_batch_size": 32,  # decrease for larger models
    },
)
```

**Prerequisites:**
```bash
pip install autogluon.multimodal
mim install mmcv
pip install "mmdet==3.1.0"
```

## Evaluation

```python
test_df = from_coco(test_path)
predictor.evaluate(test_df)
```