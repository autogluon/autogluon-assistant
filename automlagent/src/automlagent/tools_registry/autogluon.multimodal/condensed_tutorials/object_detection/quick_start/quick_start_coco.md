# Condensed: AutoMM Detection - Quick Start on a Tiny COCO Format Dataset

Summary: This tutorial demonstrates implementing object detection using AutoGluon's MultiModalPredictor, covering setup requirements, model configuration, and inference workflows. It provides implementation details for training and evaluating models with different quality presets (YOLOX-large, DINO-Resnet50, DINO-SwinL), handling COCO-format datasets, and performing predictions with confidence thresholds. Key functionalities include model saving/loading, GPU configuration, visualization tools, and working with both single images and COCO-format data. The tutorial is particularly useful for tasks involving object detection model setup, training pipelines, and inference optimization with AutoGluon's framework.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on key implementation details and practices:

# AutoMM Detection Quick Start Guide

## Key Setup Requirements
```python
# Essential installations
pip install autogluon.multimodal
python3 -m mim install "mmcv==2.1.0"
python3 -m pip install "mmdet==3.2.0"
python3 -m pip install "mmengine>=0.10.6"

# Core imports
from autogluon.multimodal import MultiModalPredictor
```

## Important Notes
- MMDet requires MMCV 2.1.0
- Best compatibility: CUDA 12.4 + PyTorch 2.5
- Install build dependencies before MMCV: `pip install -U pip setuptools wheel`

## Dataset Requirements
- COCO format JSON files:
  - `trainval_cocoformat.json`: training/validation data
  - `test_cocoformat.json`: test data

## Model Configuration

```python
predictor = MultiModalPredictor(
    problem_type="object_detection",
    sample_data_path=train_path,
    presets="medium_quality",  # Options: medium_quality, high_quality, best_quality
    path=model_path  # Optional save location
)
```

### Preset Options
- `medium_quality`: YOLOX-large (fast, efficient)
- `high_quality`: DINO-Resnet50 (better accuracy)
- `best_quality`: DINO-SwinL (highest accuracy, slower)

## Training and Evaluation
```python
# Training
predictor.fit(train_path)

# Evaluation
predictor.evaluate(test_path)

# Load saved model
new_predictor = MultiModalPredictor.load(model_path)
new_predictor.set_num_gpus(1)  # Adjust GPU usage
```

## Inference
```python
# Predict with confidence threshold
pred = predictor.predict(test_path, save_results=True, as_coco=False)

# Visualization
from autogluon.multimodal.utils import ObjectDetectionVisualizer

visualizer = ObjectDetectionVisualizer(img_path)
out = visualizer.draw_instance_predictions(image_result, conf_threshold=0.4)
```

## Output Format
Predictions returned as DataFrame with:
- `image`: Image path
- `bboxes`: List of detections
  ```python
  {
      "class": "class_name",
      "bbox": [x1, y1, x2, y2],
      "score": confidence_score
  }
  ```

## Best Practices
1. Use `medium_quality` preset for quick prototyping
2. Switch to `high_quality` or `best_quality` for better accuracy
3. Save predictions with `save_results=True`
4. Set confidence threshold for visualization
5. Restart kernel after MMCV installation

## Custom Data Inference
```python
# Single image
predictor.predict([image_path])

# COCO format
predictor.predict(json_annotation_file)
```

This condensed version maintains all critical implementation details while removing explanatory text and redundant examples.