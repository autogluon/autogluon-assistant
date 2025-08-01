# Condensed: To start, make sure `mmcv` and `mmdet` are installed.

Summary: This tutorial demonstrates implementing object detection with AutoGluon's MultiModalPredictor. It covers installation requirements, dataset preparation, model configuration with three quality presets (YOLOX-large, DINO-Resnet50, DINO-SwinL), training and evaluation workflows, model saving/loading, and inference techniques. Key functionalities include prediction in both CSV and COCO formats, visualization of detection results using OpenCV, and custom inference on new images through either COCO JSON format or direct image path lists. The tutorial provides practical code for implementing complete object detection pipelines with performance optimization options.

*This is a condensed version that preserves essential implementation details and context.*

# Object Detection with AutoMM

## Installation

```bash
# Install dependencies
pip install -U pip setuptools wheel
sudo apt-get install -y ninja-build gcc g++

# Install required packages
pip install autogluon.multimodal
python3 -m mim install "mmcv==2.1.0"
python3 -m pip install "mmdet==3.2.0"
python3 -m pip install "mmengine>=0.10.6"
```

**Note:** MMDet is only compatible with MMCV 2.1.0. For best results, use CUDA 12.4 with PyTorch 2.5.

## Setup

```python
from autogluon.multimodal import MultiModalPredictor
import os
import time
from autogluon.core.utils.loaders import load_zip

# Download dataset
zip_file = "https://automl-mm-bench.s3.amazonaws.com/object_detection_dataset/tiny_motorbike_coco.zip"
download_dir = "./tiny_motorbike_coco"
load_zip.unzip(zip_file, unzip_dir=download_dir)
data_dir = os.path.join(download_dir, "tiny_motorbike")
train_path = os.path.join(data_dir, "Annotations", "trainval_cocoformat.json")
test_path = os.path.join(data_dir, "Annotations", "test_cocoformat.json")
```

## Model Configuration

```python
# Set model path
model_path = f"./tmp/{uuid.uuid4().hex}-quick_start_tutorial_temp_save"

# Initialize predictor
predictor = MultiModalPredictor(
    problem_type="object_detection",
    sample_data_path=train_path,
    presets="medium_quality",  # Uses YOLOX-large pretrained on COCO
    path=model_path,
)
```

**Available presets:**
- `medium_quality`: YOLOX-large (fast training, quick inference)
- `high_quality`: DINO-Resnet50 (better performance, slower)
- `best_quality`: DINO-SwinL (highest performance, resource intensive)

## Training and Evaluation

```python
# Train model
start = time.time()
predictor.fit(train_path)
train_end = time.time()
print(f"Training time: {train_end - start:.2f} seconds")

# Evaluate model
predictor.evaluate(test_path)
eval_end = time.time()
print(f"Evaluation time: {eval_end - train_end:.2f} seconds")
```

## Loading a Saved Model

```python
# Load model and set GPU count
new_predictor = MultiModalPredictor.load(model_path)
new_predictor.set_num_gpus(1)
```

## Inference

```python
# Run prediction
pred = predictor.predict(test_path)
print(len(pred))  # Number of predictions
print(pred[:3])   # Sample of first 3 predictions

# Save predictions
pred = predictor.predict(test_path, save_results=True, as_coco=False)  # CSV format
pred = predictor.predict(test_path, save_results=True, as_coco=True, result_save_path="./results.json")  # COCO format
```

**Output format:** DataFrame with columns:
- `image`: Path to input image
- `bboxes`: List of detected objects with format:
  ```
  {
      "class": "predicted_class_name",
      "bbox": [x1, y1, x2, y2],  # Coordinates of corners
      "score": confidence_score
  }
  ```

## Visualization

```python
# Install OpenCV if needed
pip install opencv-python

# Visualize detection results
from autogluon.multimodal.utils import ObjectDetectionVisualizer
from PIL import Image
from IPython.display import display

conf_threshold = 0.4
image_result = pred.iloc[30]
img_path = image_result.image

visualizer = ObjectDetectionVisualizer(img_path)
out = visualizer.draw_instance_predictions(image_result, conf_threshold=conf_threshold)
visualized = out.get_image()

img = Image.fromarray(visualized, 'RGB')
display(img)
```

## Custom Data Inference

```python
# Download test image
from autogluon.multimodal.utils import download
image_url = "https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/detection/street_small.jpg"
test_image = download(image_url)

# Method 1: Using COCO format JSON
import json
data = {"images": [{"id": 0, "width": -1, "height": -1, "file_name": test_image}], "categories": []}
os.mkdir("input_data_for_demo")
input_file = "input_data_for_demo/demo_annotation.json"
with open(input_file, "w+") as f:
    json.dump(data, f)
pred_test_image = predictor.predict(input_file)

# Method 2: Using list of image paths
pred_test_image = predictor.predict([test_image])
```