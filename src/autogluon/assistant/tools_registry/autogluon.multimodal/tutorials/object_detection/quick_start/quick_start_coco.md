Summary: This tutorial demonstrates implementing object detection using AutoGluon's MultiModalPredictor, covering setup requirements, model configuration, and inference workflows. It provides implementation details for training and evaluating models with different quality presets (YOLOX-large, DINO-Resnet50, DINO-SwinL), handling COCO-format datasets, and performing predictions with confidence thresholds. Key functionalities include model saving/loading, GPU configuration, visualization tools, and working with both single images and COCO-format data. The tutorial is particularly useful for tasks involving object detection model setup, training pipelines, and inference optimization with AutoGluon's framework.

# AutoMM Detection - Quick Start on a Tiny COCO Format Dataset

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/autogluon/autogluon/blob/master/docs/tutorials/multimodal/object_detection/quick_start/quick_start_coco.ipynb)
[![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/autogluon/autogluon/blob/master/docs/tutorials/multimodal/object_detection/quick_start/quick_start_coco.ipynb)



In this section, our goal is to fast finetune a pretrained model on a small dataset in COCO format, 
and evaluate on its test set. Both training and test sets are in COCO format.
See [Convert Data to COCO Format](../data_preparation/convert_data_to_coco_format.ipynb) for how to convert other datasets to COCO format.

## Setting up the imports

To start, make sure `mmcv` and `mmdet` are installed.
**Note:** MMDet is no longer actively maintained and is only compatible with MMCV version 2.1.0. Installation can be problematic due to CUDA version compatibility issues. For best results:
1. Use CUDA 12.4 with PyTorch 2.5
2. Before installation, run:
   ```bash
   pip install -U pip setuptools wheel
   sudo apt-get install -y ninja-build gcc g++
   ```
   This will help prevent MMCV installation from hanging during wheel building.
3. After installation in Jupyter notebook, restart the kernel for changes to take effect.



```python
!pip install autogluon.multimodal
```


```python
# Update package tools and install build dependencies
!pip install -U pip setuptools wheel
!sudo apt-get install -y ninja-build gcc g++

# Install MMCV
!python3 -m mim install "mmcv==2.1.0"

# For Google Colab users: If the above fails, use this alternative MMCV installation
# pip install "mmcv==2.1.0" -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html

# Install MMDet
!python3 -m pip install "mmdet==3.2.0"

# Install MMEngine (version >=0.10.6 for PyTorch 2.5 compatibility)
!python3 -m pip install "mmengine>=0.10.6"
```

To start, let's import MultiModalPredictor:


```python
from autogluon.multimodal import MultiModalPredictor
```

And also import some other packages that will be used in this tutorial:


```python
import os
import time

from autogluon.core.utils.loaders import load_zip
```

## Downloading Data
We have the sample dataset ready in the cloud. Let's download it:


```python
zip_file = "https://automl-mm-bench.s3.amazonaws.com/object_detection_dataset/tiny_motorbike_coco.zip"
download_dir = "./tiny_motorbike_coco"

load_zip.unzip(zip_file, unzip_dir=download_dir)
data_dir = os.path.join(download_dir, "tiny_motorbike")
train_path = os.path.join(data_dir, "Annotations", "trainval_cocoformat.json")
test_path = os.path.join(data_dir, "Annotations", "test_cocoformat.json")
```

### Dataset Format

For COCO format datasets, provide JSON annotation files for each split:

- `trainval_cocoformat.json`: train and validation data
- `test_cocoformat.json`: test data

### Model Selection

We use the `medium_quality` preset which features:

- Base model: YOLOX-large (pretrained on COCO)
- Benefits: Fast finetuning, quick inference, easy deployment

Alternative presets available:

- `high_quality`: DINO-Resnet50 model
- `best_quality`: DINO-SwinL model

Both alternatives offer improved performance at the cost of slower processing and higher GPU memory requirements.


```python
presets = "medium_quality"
```

When creating the MultiModalPredictor, specify these essential parameters:

- `problem_type="object_detection"` to define the task
- `presets="medium_quality"` for presets selection
- `sample_data_path` pointing to any dataset split (typically train_path) to infer object categories
- `path` (optional) to set a custom save location

If no path is specified, the model will be automatically saved to a timestamped directory under AutogluonModels/.


```python
# Init predictor
import uuid

model_path = f"./tmp/{uuid.uuid4().hex}-quick_start_tutorial_temp_save"

predictor = MultiModalPredictor(
    problem_type="object_detection",
    sample_data_path=train_path,
    presets=presets,
    path=model_path,
)
```

## Finetuning the Model
The model uses optimized preset configurations for learning rate, epochs, and batch size. By default, it employs a two-stage learning rate strategy:

Model head layers use 100x higher learning rate
This approach accelerates convergence and typically improves performance, especially for small datasets (hundreds to thousands of images)

Timing results below are from a test run on AWS g4.2xlarge EC2 instance:


```python
start = time.time()
predictor.fit(train_path)  # Fit
train_end = time.time()
```

Notice that at the end of each progress bar, if the checkpoint at current stage is saved,
it prints the model's save path.
In this example, it's `./quick_start_tutorial_temp_save`.

Print out the time and we can see that it's fast!


```python
print("This finetuning takes %.2f seconds." % (train_end - start))
```

## Evaluation

To evaluate the model we just trained, run following code.

And the evaluation results are shown in command line output. 
The first line is mAP in COCO standard, and the second line is mAP in VOC standard (or mAP50). 
For more details about these metrics, see [COCO's evaluation guideline](https://cocodataset.org/#detection-eval).
Note that for presenting a fast finetuning we use presets "medium_quality", 
you could get better result on this dataset by simply using "high_quality" or "best_quality" presets, 
or customize your own model and hyperparameter settings: [Customization](../../advanced_topics/customization.ipynb), and some other examples at [Fast Fine-tune Coco](../finetune/detection_fast_finetune_coco) or [High Performance Fine-tune Coco](../finetune/detection_high_performance_finetune_coco).


```python
predictor.evaluate(test_path)
eval_end = time.time()
```

Print out the evaluation time:


```python
print("The evaluation takes %.2f seconds." % (eval_end - train_end))
```

We can load a new predictor with previous save path,
and we can also reset the number of used GPUs if not all the devices are available:


```python
# Load and reset num_gpus
new_predictor = MultiModalPredictor.load(model_path)
new_predictor.set_num_gpus(1)
```

Evaluating the new predictor gives us exactly the same result:


```python
# Evaluate new predictor
new_predictor.evaluate(test_path)
```

For how to set the hyperparameters and finetune the model with higher performance, 
see [AutoMM Detection - High Performance Finetune on COCO Format Dataset](../finetune/detection_high_performance_finetune_coco.ipynb).

## Inference
Let's perform predictions using our finetuned model. The predictor can process the entire test set with a single command:


```python
pred = predictor.predict(test_path)
print(len(pred))  # Number of predictions
print(pred[:3])   # Sample of first 3 predictions
```

The predictor returns predictions as a pandas DataFrame with two columns:
- `image`: Contains path to each input image
- `bboxes`: Contains list of detected objects, where each object is a dictionary:
  ```python
  {
      "class": "predicted_class_name",
      "bbox": [x1, y1, x2, y2],  # Coordinates of Upper Left and Bottom Right corners
      "score": confidence_score
  }
  ```

By default, predictions are returned but not saved. To save detection results, use the save parameter in your predict call.


```python
pred = predictor.predict(test_path, save_results=True, as_coco=False)
```

The predictions can be saved in two formats:

- CSV file: Matches the DataFrame structure with image and bboxes columns
- COCO JSON: Standard COCO format annotation file

This works with any predictor configuration (pretrained or finetuned models).

## Visualizing Results
To run visualizations, ensure that you have `opencv` installed. If you haven't already, install `opencv` by running


```python
!pip install opencv-python
```

To visualize the detection bounding boxes, run the following:


```python
from autogluon.multimodal.utils import ObjectDetectionVisualizer

conf_threshold = 0.4  # Specify a confidence threshold to filter out unwanted boxes
image_result = pred.iloc[30]

img_path = image_result.image  # Select an image to visualize

visualizer = ObjectDetectionVisualizer(img_path)  # Initialize the Visualizer
out = visualizer.draw_instance_predictions(image_result, conf_threshold=conf_threshold)  # Draw detections
visualized = out.get_image()  # Get the visualized image

from PIL import Image
from IPython.display import display
img = Image.fromarray(visualized, 'RGB')
display(img)
```

## Testing on Your Own Data
You can also predict on your own images with various input format. The follow is an example:

Download the example image:


```python
from autogluon.multimodal import download
image_url = "https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/detection/street_small.jpg"
test_image = download(image_url)
```

Run inference on data in a json file of COCO format (See [Convert Data to COCO Format](../data_preparation/convert_data_to_coco_format.ipynb) for more details about COCO format). Note that since the root is by default the parent folder of the annotation file, here we put the annotation file in a folder:


```python
import json

# create a input file for demo
data = {"images": [{"id": 0, "width": -1, "height": -1, "file_name": test_image}], "categories": []}
os.mkdir("input_data_for_demo")
input_file = "input_data_for_demo/demo_annotation.json"
with open(input_file, "w+") as f:
    json.dump(data, f)

pred_test_image = predictor.predict(input_file)
print(pred_test_image)
```

Run inference on data in a list of image file names:


```python
pred_test_image = predictor.predict([test_image])
print(pred_test_image)
```

## Other Examples

You may go to [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm) to explore other examples about AutoMM.

## Customization
To learn how to customize AutoMM, please refer to [Customize AutoMM](../../advanced_topics/customization.ipynb).

## Citation

```
@article{DBLP:journals/corr/abs-2107-08430,
  author    = {Zheng Ge and
               Songtao Liu and
               Feng Wang and
               Zeming Li and
               Jian Sun},
  title     = {{YOLOX:} Exceeding {YOLO} Series in 2021},
  journal   = {CoRR},
  volume    = {abs/2107.08430},
  year      = {2021},
  url       = {https://arxiv.org/abs/2107.08430},
  eprinttype = {arXiv},
  eprint    = {2107.08430},
  timestamp = {Tue, 05 Apr 2022 14:09:44 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2107-08430.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org},
}
```

