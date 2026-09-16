# Condensed: [TensorRT](https://developer.nvidia.com/tensorrt), built on the NVIDIA CUDA® parallel programming model, enables us to optimize inference by leveraging libraries, development tools, and technologies in NVIDIA AI, autonomous machines, high-performance computing, and graphics. AutoGluon-MultiModal is now integrated with TensorRT via `predictor.optimize_for_inference()` interface. This tutorial demonstates how to leverage TensorRT in boosting inference speed, which would be helpful in increasing efficiency at deployment environment.

Summary: This tutorial demonstrates TensorRT integration with AutoGluon-MultiModal for accelerating multimodal model inference. It covers: (1) installing TensorRT/ONNX dependencies, (2) training MultiModalPredictor with mixed data types (images, text, tabular), (3) applying `optimize_for_inference()` to convert PyTorch models to ONNX Runtime for faster inference, (4) benchmarking speed improvements, and (5) validating accuracy with FP16 mixed precision. Key techniques include proper model loading/reloading workflow (since optimization is inference-only), configuring execution providers for precision control, and comparing PyTorch vs TensorRT performance. Useful for deployment optimization tasks requiring faster inference on multimodal classification/regression models while maintaining accuracy within acceptable tolerances (atol=0.01).

*This is a condensed version that preserves essential implementation details and context.*

# TensorRT Optimization for AutoGluon-MultiModal

AutoGluon-MultiModal integrates with [TensorRT](https://developer.nvidia.com/tensorrt) via `predictor.optimize_for_inference()` to boost inference speed at deployment.

## Setup

### Install Dependencies
TensorRT/ONNX/ONNXRUNTIME-GPU are optional dependencies:

```python
try:
    import tensorrt, onnx, onnxruntime
except ImportError:
    !pip install autogluon.multimodal[tests]
    !pip install -U "tensorrt>=10.0.0b0,<11.0"
```

### Dataset Preparation
Using PetFinder dataset (binary classification: adoption speed):

```python
from autogluon.core.utils.loaders import load_zip
load_zip.unzip('https://automl-mm-bench.s3.amazonaws.com/petfinder_for_tutorial.zip', 
               unzip_dir='./ag_automm_tutorial')

train_data = pd.read_csv(f'{dataset_path}/train.csv', index_col=0)
test_data = pd.read_csv(f'{dataset_path}/test.csv', index_col=0)

# Expand image paths
train_data[image_col] = train_data[image_col].apply(
    lambda ele: path_expander(ele.split(';')[0], base_folder=dataset_path))
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
    time_limit=120
)
```

## Inference Optimization

### Standard PyTorch Inference
```python
batch_size = 2
y_pred = predictor.predict_proba(sample)  # Warmup
# Benchmark predictions...
```

### TensorRT-Optimized Inference
```python
trt_predictor = MultiModalPredictor.load(path=predictor.path)
trt_predictor.optimize_for_inference()
y_pred_trt = trt_predictor.predict_proba(sample)
```

**Key Implementation:** `optimize_for_inference()` replaces the internal torch module with an ONNX Runtime-based module for optimized inference.

⚠️ **Warning:** `optimize_for_inference()` modifies the model for inference only. Calling `predictor.fit()` afterward will error. Reload with `MultiModalPredictor.load()` to refit.

## Validation

### Accuracy Comparison
Uses mixed precision (FP16) by default, causing minor differences:

```python
np.testing.assert_allclose(y_pred, y_pred_trt, atol=0.01)

# Evaluate on full test set
metric = predictor.evaluate(test_data)
metric_trt = trt_predictor.evaluate(test_data)
```

Results should be very close. If significant gaps occur, disable mixed precision:

```python
predictor.optimize_for_inference(providers=["CUDAExecutionProvider"])
```

See [Execution Providers](https://onnxruntime.ai/docs/execution-providers/) for full provider list and [TensorRT Reduced Precision Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#reduced-precision) for precision details.

### Speed Visualization
```python
infer_speed = batch_size/np.mean(pred_time)
infer_speed_trt = batch_size/np.mean(pred_time_trt)

# Visualize with bar chart comparing PyTorch vs TensorRT throughput
```

## Additional Resources
- [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm)
- [Customize AutoMM](customization.ipynb)