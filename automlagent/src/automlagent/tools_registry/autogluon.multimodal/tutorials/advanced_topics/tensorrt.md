Summary: This tutorial demonstrates how to optimize AutoGluon models using TensorRT for faster inference while maintaining accuracy. It covers implementation techniques for model optimization using optimize_for_inference(), including proper setup with CUDA providers, batch size considerations, and precision settings (FP16). The tutorial helps with tasks like converting trained MultiModalPredictor models to TensorRT-optimized versions, validating optimization results, and handling post-optimization workflows. Key features include seamless TensorRT integration, performance validation methods, mixed precision support, and important warnings about model modification constraints and retraining procedures.

# Faster Prediction with TensorRT

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/autogluon/autogluon/blob/master/docs/tutorials/multimodal/advanced_topics/tensorrt.ipynb)
[![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/autogluon/autogluon/blob/master/docs/tutorials/multimodal/advanced_topics/tensorrt.ipynb)

[TensorRT](https://developer.nvidia.com/tensorrt), built on the NVIDIA CUDA® parallel programming model, enables us to optimize inference by leveraging libraries, development tools, and technologies in NVIDIA AI, autonomous machines, high-performance computing, and graphics. AutoGluon-MultiModal is now integrated with TensorRT via `predictor.optimize_for_inference()` interface. This tutorial demonstates how to leverage TensorRT in boosting inference speed, which would be helpful in increasing efficiency at deployment environment.


```python
import os
import numpy as np
import time
import warnings
from IPython.display import clear_output
warnings.filterwarnings('ignore')
np.random.seed(123)
```

### Install required packages
Since the tensorrt/onnx/onnxruntime-gpu packages are currently optional dependencies of autogluon.multimodal, we need to ensure these packages are correctly installed.


```python
try:
    import tensorrt, onnx, onnxruntime
    print(f"tensorrt=={tensorrt.__version__}, onnx=={onnx.__version__}, onnxruntime=={onnxruntime.__version__}")
except ImportError:
    !pip install autogluon.multimodal[tests]
    !pip install -U "tensorrt>=10.0.0b0,<11.0"
    clear_output()
```

## Dataset

For demonstration, we use a simplified and subsampled version of [PetFinder dataset](https://www.kaggle.com/c/petfinder-adoption-prediction). The task is to predict the animals' adoption rates based on their adoption profile information. In this simplified version, the adoption speed is grouped into two categories: 0 (slow) and 1 (fast).

To get started, let's download and prepare the dataset.


```python
download_dir = './ag_automm_tutorial'
zip_file = 'https://automl-mm-bench.s3.amazonaws.com/petfinder_for_tutorial.zip'
from autogluon.core.utils.loaders import load_zip
load_zip.unzip(zip_file, unzip_dir=download_dir)
```

Next, we will load the CSV files.


```python
import pandas as pd
dataset_path = download_dir + '/petfinder_for_tutorial'
train_data = pd.read_csv(f'{dataset_path}/train.csv', index_col=0)
test_data = pd.read_csv(f'{dataset_path}/test.csv', index_col=0)
label_col = 'AdoptionSpeed'
```

We need to expand the image paths to load them in training.


```python
image_col = 'Images'
train_data[image_col] = train_data[image_col].apply(lambda ele: ele.split(';')[0]) # Use the first image for a quick tutorial
test_data[image_col] = test_data[image_col].apply(lambda ele: ele.split(';')[0])

def path_expander(path, base_folder):
    path_l = path.split(';')
    return ';'.join([os.path.abspath(os.path.join(base_folder, path)) for path in path_l])

train_data[image_col] = train_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
test_data[image_col] = test_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
```

Each animal's adoption profile includes pictures, a text description, and various tabular features such as age, breed, name, color, and more.

## Training
Now let's fit the predictor with the training data. Here we set a tight time budget for a quick demo.


```python
from autogluon.multimodal import MultiModalPredictor
hyperparameters = {
    "optimization.max_epochs": 2,
    "model.names": ["numerical_mlp", "categorical_mlp", "timm_image", "hf_text", "fusion_mlp"],
    "model.timm_image.checkpoint_name": "mobilenetv3_small_100",
    "model.hf_text.checkpoint_name": "google/electra-small-discriminator",
    
}
predictor = MultiModalPredictor(label=label_col).fit(
    train_data=train_data,
    hyperparameters=hyperparameters,
    time_limit=120, # seconds
)

clear_output()
```

Under the hood, AutoMM automatically infers the problem type (classification or regression), detects the data modalities, selects the related models from the multimodal model pools, and trains the selected models. If multiple backbones are available, AutoMM appends a late-fusion model (MLP or transformer) on top of them.

## Prediction with default PyTorch module
Given a multimodal dataframe without the label column, we can predict the labels.

Note that we would use a small sample of test data here for benchmarking. Later, we would evaluate over the whole test dataset to assess accuracy loss.


```python
batch_size = 2
n_trails = 10
sample = test_data.head(batch_size)

# Use first prediction for initialization (e.g., allocating memory)
y_pred = predictor.predict_proba(sample)

pred_time = []
for _ in range(n_trails):
    tic = time.time()
    y_pred = predictor.predict_proba(sample)
    elapsed = time.time()-tic
    pred_time.append(elapsed)
    print(f"elapsed (pytorch): {elapsed*1000:.1f} ms (batch_size={batch_size})")
```

## Prediction with TensorRT module

First, let's load a new predictor that optimize it for inference.


```python
model_path = predictor.path
trt_predictor = MultiModalPredictor.load(path=model_path)
trt_predictor.optimize_for_inference()

# Again, use first prediction for initialization (e.g., allocating memory)
y_pred_trt = trt_predictor.predict_proba(sample)

clear_output()
```

Under the hood, the `optimize_for_inference()` would generate an onnxruntime-based module that can be a drop-in replacement of torch.nn.Module. It would replace the internal torch-based module `predictor._model` for optimized inference.

```{warning}
The function `optimize_for_inference()` would modify internal model definition for inference only. Calling `predictor.fit()` after this would result in an error.
It is recommended to reload the model with `MultiModalPredictor.load`, in order to refit the model.
```

Then, we can perform prediction or extract embeddings as usual. For fair inference speed comparison, here we run prediction multiple times.


```python
pred_time_trt = []
for _ in range(n_trails):
    tic = time.time()
    y_pred_trt = trt_predictor.predict_proba(sample)
    elapsed = time.time()-tic
    pred_time_trt.append(elapsed)
    print(f"elapsed (tensorrt): {elapsed*1000:.1f} ms (batch_size={batch_size})")
```

To verify the correctness of the prediction results, we can compare the results side-by-side.

Let's take a peek at the expected results and TensorRT results.


```python
y_pred, y_pred_trt
```

As we are using mixed precision (FP16) by default, there might be loss of accuracy. We can see the probabilities are quite close, and we should be able to safely assume these results are relatively close for most of the cases. Refer to [Reduced Precision section in TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#reduced-precision) for more details.


```python
np.testing.assert_allclose(y_pred, y_pred_trt, atol=0.01)
```

### Visualize Inference Speed

We can calculate inference time by dividing the prediction time.


```python
infer_speed = batch_size/np.mean(pred_time)
infer_speed_trt = batch_size/np.mean(pred_time_trt)
```

Then, visualize speed improvements.


```python
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
fig.set_figheight(1.5)
ax.barh(["PyTorch", "TensorRT"], [infer_speed, infer_speed_trt])
ax.annotate(f"{infer_speed:.1f} rows/s", xy=(infer_speed, 0))
ax.annotate(f"{infer_speed_trt:.1f} rows/s", xy=(infer_speed_trt, 1))
_ = plt.xlabel('Inference Speed (rows per second)')
```

### Compare Evaluation Metric
Now that we can achieve better inference speed with `optimize_for_inference()`, but is there any impact to the underlining accuracy loss?

Let's start with whole test dataset evaluation.


```python
metric = predictor.evaluate(test_data)
metric_trt = trt_predictor.evaluate(test_data)
clear_output()
```


```python
metric_df = pd.DataFrame.from_dict({"PyTorch": metric, "TensorRT": metric_trt})
metric_df
```

The evaluation results are expected to be very close.

In case there is any significant gap between the evaluation results, try disabling mixed precision by using CUDA execution provider:

```python
predictor.optimize_for_inference(providers=["CUDAExecutionProvider"])
```

See [Execution Providers](https://onnxruntime.ai/docs/execution-providers/) for a full list of providers.

## Other Examples

You may go to [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm) to explore other examples about AutoMM.

## Customization
To learn how to customize AutoMM, please refer to [Customize AutoMM](customization.ipynb).
