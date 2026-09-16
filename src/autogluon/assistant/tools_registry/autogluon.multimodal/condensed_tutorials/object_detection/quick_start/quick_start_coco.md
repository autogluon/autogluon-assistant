# Condensed: To start, make sure `mmcv` and `mmdet` are installed.

Summary: This tutorial covers AutoGluon MultiModal object detection implementation using YOLOX-large and DINO models with COCO format datasets. It provides code for installing MMDet/MMCV dependencies with specific version requirements (MMCV 2.1.0, CUDA 12.4, PyTorch 2.5), configuring MultiModalPredictor with three preset quality levels (medium/high/best), and implementing the complete workflow: data loading, model training with two-stage learning rate strategy, evaluation using mAP metrics, inference on COCO JSON or image lists, and visualization with bounding boxes. Key functionalities include saving/loading models, adjusting GPU settings, exporting predictions as CSV or COCO JSON format, and handling custom datasets. The tutorial helps with tasks like fine-tuning pretrained object detectors, processing COCO annotations, batch prediction, result visualization, and model deployment.

*This is a condensed version that preserves essential implementation details and context.*

# Object Detection with AutoGluon MultiModal - Quick Start

## Installation

**Critical:** MMDet is no longer actively maintained and requires MMCV 2.1.0. Use CUDA 12.4 with PyTorch 2.5 for best compatibility.

```bash
# Update package tools and install build dependencies
pip install -U pip setuptools wheel
sudo apt-get install -y ninja-build gcc g++

# Install required packages
pip install autogluon.multimodal
python3 -m mim install "mmcv==2.1.0"
python3 -m pip install "mmdet==3.2.0"
python3 -m pip install "mmengine>=0.10.6"

# Alternative for Google Colab if above fails:
# pip install "mmcv==2.1.0" -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html
```

**Note:** Restart Jupyter kernel after installation.

## Setup and Data Preparation

```python
from autogluon.multimodal import MultiModalPredictor
from autogluon.core.utils.loaders import load_zip
import os

# Download dataset
zip_file = "https://automl-mm-bench.s3.amazonaws.com/object_detection_dataset/tiny_motorbike_coco.zip"
download_dir = "./tiny_motorbike_coco"
load_zip.unzip(zip_file, unzip_dir=download_dir)

data_dir = os.path.join(download_dir, "tiny_motorbike")
train_path = os.path.join(data_dir, "Annotations", "trainval_cocoformat.json")
test_path = os.path.join(data_dir, "Annotations", "test_cocoformat.json")
```

**Dataset Format:** Provide COCO format JSON annotation files for each split.

## Model Configuration

```python
import uuid

presets = "medium_quality"  # YOLOX-large: fast finetuning, quick inference
model_path = f"./tmp/{uuid.uuid4().hex}-quick_start_tutorial_temp_save"

predictor = MultiModalPredictor(
    problem_type="object_detection",
    sample_data_path=train_path,  # Infers object categories
    presets=presets,
    path=model_path,  # Optional: custom save location
)
```

**Preset Options:**
- `medium_quality`: YOLOX-large (fast, efficient)
- `high_quality`: DINO-Resnet50 (better performance, slower)
- `best_quality`: DINO-SwinL (best performance, highest resource requirements)

## Training

```python
predictor.fit(train_path)
```

**Key Features:**
- Two-stage learning rate strategy: model head uses 100x higher learning rate
- Accelerates convergence, especially for small datasets (hundreds to thousands of images)
- Optimized preset configurations for learning rate, epochs, and batch size

## Evaluation

```python
predictor.evaluate(test_path)
```

**Metrics:** Returns mAP in COCO standard (first line) and VOC standard/mAP50 (second line). See [COCO evaluation guidelines](https://cocodataset.org/#detection-eval).

## Loading Saved Models

```python
new_predictor = MultiModalPredictor.load(model_path)
new_predictor.set_num_gpus(1)  # Reset GPU count if needed
new_predictor.evaluate(test_path)
```

## Inference

```python
# Basic prediction
pred = predictor.predict(test_path)

# Save results as CSV
pred = predictor.predict(test_path, save_results=True, as_coco=False)

# Save results as COCO JSON
pred = predictor.predict(test_path, save_results=True, as_coco=True, 
                        result_save_path="./results.json")
```

**Output Format:** DataFrame with columns:
- `image`: Path to input image
- `bboxes`: List of detected objects:
  ```python
  {
      "class": "predicted_class_name",
      "bbox": [x1, y1, x2, y2],  # Upper Left and Bottom Right corners
      "score": confidence_score
  }
  ```

## Visualization

```python
pip install opencv-python

from autogluon.multimodal.utils import ObjectDetectionVisualizer
from PIL import Image
from IPython.display import display

conf_threshold = 0.4
image_result = pred.iloc[30]

visualizer = ObjectDetectionVisualizer(image_result.image)
out = visualizer.draw_instance_predictions(image_result, conf_threshold=conf_threshold)
visualized = out.get_image()

img = Image.fromarray(visualized, 'RGB')
display(img)
```

## Custom Data Inference

**COCO JSON format:**
```python
import json

data = {"images": [{"id": 0, "width": -1, "height": -1, "file_name": test_image}], 
        "categories": []}
os.mkdir("input_data_for_demo")
input_file = "input_data_for_demo/demo_annotation.json"
with open(input_file, "w+") as f:
    json.dump(data, f)

pred = predictor.predict(input_file)
```

**List of image paths:**
```python
pred = predictor.predict([test_image])
```

## Additional Resources

- [High Performance Finetune on COCO](../finetune/detection_high_performance_finetune_coco.ipynb)
- [Convert Data to COCO Format](../data_preparation/convert_data_to_coco_format.ipynb)
- [Customize AutoMM](../../advanced_topics/customization.ipynb)
- [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm)