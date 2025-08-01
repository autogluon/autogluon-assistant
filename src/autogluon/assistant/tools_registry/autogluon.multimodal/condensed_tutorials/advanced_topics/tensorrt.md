# Condensed: [TensorRT](https://developer.nvidia.com/tensorrt), built on the NVIDIA CUDA® parallel programming model, enables us to optimize inference by leveraging libraries, development tools, and technologies in NVIDIA AI, autonomous machines, high-performance computing, and graphics. AutoGluon-MultiModal is now integrated with TensorRT via `predictor.optimize_for_inference()` interface. This tutorial demonstates how to leverage TensorRT in boosting inference speed, which would be helpful in increasing efficiency at deployment environment.

Summary: This tutorial demonstrates integrating TensorRT with AutoGluon-MultiModal to optimize inference speed. It covers: (1) implementing TensorRT optimization for multimodal models through the optimize_for_inference() method, (2) accelerating prediction tasks for multimodal classification with image and text data, and (3) key functionalities including model training with MultiModalPredictor, converting PyTorch models to TensorRT, benchmarking inference performance, and configuring execution providers. The tutorial shows how to verify prediction consistency while achieving significant speedups, with warnings about post-optimization limitations.

*This is a condensed version that preserves essential implementation details and context.*

# TensorRT Integration with AutoGluon-MultiModal

This tutorial demonstrates how to use TensorRT to optimize inference speed in AutoGluon-MultiModal.

## Setup

```python
import os
import numpy as np
import time
import warnings
from IPython.display import clear_output
warnings.filterwarnings('ignore')
np.random.seed(123)

# Install required packages if needed
try:
    import tensorrt, onnx, onnxruntime
    print(f"tensorrt=={tensorrt.__version__}, onnx=={onnx.__version__}, onnxruntime=={onnxruntime.__version__}")
except ImportError:
    !pip install autogluon.multimodal[tests]
    !pip install -U "tensorrt>=10.0.0b0,<11.0"
```

## Dataset Preparation

```python
# Download and prepare dataset
download_dir = './ag_automm_tutorial'
zip_file = 'https://automl-mm-bench.s3.amazonaws.com/petfinder_for_tutorial.zip'
from autogluon.core.utils.loaders import load_zip
load_zip.unzip(zip_file, unzip_dir=download_dir)

# Load CSV files
import pandas as pd
dataset_path = download_dir + '/petfinder_for_tutorial'
train_data = pd.read_csv(f'{dataset_path}/train.csv', index_col=0)
test_data = pd.read_csv(f'{dataset_path}/test.csv', index_col=0)
label_col = 'AdoptionSpeed'

# Expand image paths
image_col = 'Images'
train_data[image_col] = train_data[image_col].apply(lambda ele: ele.split(';')[0])
test_data[image_col] = test_data[image_col].apply(lambda ele: ele.split(';')[0])

def path_expander(path, base_folder):
    path_l = path.split(';')
    return ';'.join([os.path.abspath(os.path.join(base_folder, path)) for path in path_l])

train_data[image_col] = train_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
test_data[image_col] = test_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
```

## Training

```python
from autogluon.multimodal import MultiModalPredictor
hyperparameters = {
    "optim.max_epochs": 2,
    "model.names": ["numerical_mlp", "categorical_mlp", "timm_image", "hf_text", "fusion_mlp"],
    "model.timm_image.checkpoint_name": "mobilenetv3_small_100",
    "model.hf_text.checkpoint_name": "google/electra-small-discriminator",
}
predictor = MultiModalPredictor(label=label_col).fit(
    train_data=train_data,
    hyperparameters=hyperparameters,
    time_limit=120, # seconds
)
```

## Prediction with Default PyTorch Module

```python
batch_size = 2
n_trails = 10
sample = test_data.head(batch_size)

# Use first prediction for initialization
y_pred = predictor.predict_proba(sample)

pred_time = []
for _ in range(n_trails):
    tic = time.time()
    y_pred = predictor.predict_proba(sample)
    elapsed = time.time()-tic
    pred_time.append(elapsed)
    print(f"elapsed (pytorch): {elapsed*1000:.1f} ms (batch_size={batch_size})")
```

## Prediction with TensorRT Module

```python
# Load and optimize the model for inference
model_path = predictor.path
trt_predictor = MultiModalPredictor.load(path=model_path)
trt_predictor.optimize_for_inference()

# First prediction for initialization
y_pred_trt = trt_predictor.predict_proba(sample)

# Benchmark prediction speed
pred_time_trt = []
for _ in range(n_trails):
    tic = time.time()
    y_pred_trt = trt_predictor.predict_proba(sample)
    elapsed = time.time()-tic
    pred_time_trt.append(elapsed)
    print(f"elapsed (tensorrt): {elapsed*1000:.1f} ms (batch_size={batch_size})")
```

> **Warning**: The function `optimize_for_inference()` modifies internal model definition for inference only. Calling `predictor.fit()` after this will result in an error. It is recommended to reload the model with `MultiModalPredictor.load` to refit the model.

## Verify Results and Compare Performance

```python
# Verify prediction results
np.testing.assert_allclose(y_pred, y_pred_trt, atol=0.01)

# Calculate and visualize inference speed
infer_speed = batch_size/np.mean(pred_time)
infer_speed_trt = batch_size/np.mean(pred_time_trt)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
fig.set_figheight(1.5)
ax.barh(["PyTorch", "TensorRT"], [infer_speed, infer_speed_trt])
ax.annotate(f"{infer_speed:.1f} rows/s", xy=(infer_speed, 0))
ax.annotate(f"{infer_speed_trt:.1f} rows/s", xy=(infer_speed_trt, 1))
_ = plt.xlabel('Inference Speed (rows per second)')

# Compare evaluation metrics
metric = predictor.evaluate(test_data)
metric_trt = trt_predictor.evaluate(test_data)
metric_df = pd.DataFrame.from_dict({"PyTorch": metric, "TensorRT": metric_trt})
```

## Advanced Configuration

If there's a significant gap between evaluation results, try disabling mixed precision:

```python
predictor.optimize_for_inference(providers=["CUDAExecutionProvider"])
```

See [Execution Providers](https://onnxruntime.ai/docs/execution-providers/) for a full list of providers.