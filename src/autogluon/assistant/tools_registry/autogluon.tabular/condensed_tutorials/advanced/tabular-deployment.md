# Condensed: ```python

Summary: This tutorial covers AutoGluon TabularPredictor deployment workflows including training models, making predictions, and model persistence strategies. Key techniques include: (1) basic predictor training and inference with `TabularPredictor.fit()` and `.predict()`, (2) full predictor cloning via `.clone()` for experimentation snapshots, (3) deployment-optimized cloning with `.clone_for_deployment()` that reduces disk usage by keeping only inference artifacts, (4) memory optimization using `.persist()` to avoid model reloading, and (5) experimental ONNX compilation via `.compile()` for faster sklearn model inference. Helps with tasks like saving/loading trained models, creating production-ready predictor artifacts with minimal footprint, evaluating model performance with `.leaderboard()`, and optimizing inference speed through compilation and memory persistence.

*This is a condensed version that preserves essential implementation details and context.*

# AutoGluon Tabular Deployment Tutorial

## Installation & Basic Setup

```python
!pip install autogluon.tabular[all]
```

## Training a Predictor

```python
from autogluon.tabular import TabularDataset, TabularPredictor

train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
label = 'class'
subsample_size = 500  # adjust for larger datasets
train_data = train_data.sample(n=subsample_size, random_state=0)

save_path = 'agModels-predictClass-deployment'
predictor = TabularPredictor(label=label, path=save_path).fit(train_data)
```

## Making Predictions

```python
# Load predictor and test data
predictor = TabularPredictor.load(save_path)
test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')
y_test = test_data[label]

# Generate predictions
y_pred = predictor.predict(test_data)

# Evaluate models
predictor.leaderboard(test_data)
```

## Cloning Predictors

### Full Clone with `.clone()`
Creates an exact replica for experimentation while preserving the original:

```python
save_path_clone = save_path + '-clone'
path_clone = predictor.clone(path=save_path_clone)
predictor_clone = TabularPredictor.load(path=path_clone)
# Or clone and load in one step:
# predictor_clone = predictor.clone(path=save_path_clone, return_clone=True)
```

**Note:** Doubles disk usage by copying all artifacts.

### Deployment-Optimized Clone with `.clone_for_deployment()`
Creates minimal artifact set for production inference:

```python
save_path_clone_opt = save_path + '-clone-opt'
path_clone_opt = predictor.clone_for_deployment(path=save_path_clone_opt)
predictor_clone_opt = TabularPredictor.load(path=path_clone_opt)

# Persist model in memory to avoid reloading
predictor_clone_opt.persist()
```

**Limitations:** Only supports `predict()` and `predict_proba()` - cannot train additional models.

**Disk Usage Comparison:**
```python
size_original = predictor.disk_usage()
size_opt = predictor_clone_opt.disk_usage()
# Typically achieves significant reduction in disk usage
```

## Model Compilation (Experimental)

Converts sklearn models to ONNX for faster inference (RandomForest and TabularNeuralNetwork):

```python
# Install dependencies first
!pip install autogluon.tabular[skl2onnx]

# Compile models (only on cloned predictor)
predictor_clone_opt.compile()
```

**Requirements:** `skl2onnx` and `onnxruntime` packages  
**Note:** Predictions may differ slightly but remain very close to original.

## Best Practices

- **Version Consistency:** Use same Python and AutoGluon versions for training and deployment
- **Clone Before Modifications:** Always clone before operations like `.delete_models()` or `.fit_extra()`
- **Deployment:** Use `.clone_for_deployment()` for production to minimize disk usage
- **Memory Optimization:** Call `.persist()` on deployment predictors to avoid repeated model loading
- **Storage:** Upload optimized predictors to centralized storage (e.g., S3) for distribution