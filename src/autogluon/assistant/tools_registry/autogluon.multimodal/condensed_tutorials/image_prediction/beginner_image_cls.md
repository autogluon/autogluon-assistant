# Condensed: ```python

Summary: This tutorial covers AutoGluon's MultiModalPredictor for image classification tasks, demonstrating how to train models with minimal code using either image file paths or bytearrays. Key implementations include: dataset loading with `shopee_dataset()`, model training with `MultiModalPredictor.fit()` using configurable parameters (label, path, time_limit), evaluation with custom metrics, and three inference modes - `predict()` for class labels, `predict_proba()` for probability distributions, and `extract_embedding()` for feature vectors (512-2048 dimensions). The tutorial shows model persistence with auto-save/load functionality and emphasizes interchangeability between image paths and bytearrays across all operations. Useful for building image classifiers, feature extraction pipelines, and transfer learning applications with AutoMM's automated approach.

*This is a condensed version that preserves essential implementation details and context.*

# AutoMM Image Classification Tutorial (Condensed)

## Installation & Setup

```python
!pip install autogluon.multimodal
```

## Load Dataset

```python
import warnings
warnings.filterwarnings('ignore')
from autogluon.multimodal.utils.misc import shopee_dataset

download_dir = './ag_automm_tutorial_imgcls'
train_data_path, test_data_path = shopee_dataset(download_dir)
```

**Dataset format**: 2 columns - **image** (absolute paths) and **label**. 800 training rows.

### Alternative: Load as Bytearrays

```python
train_data_byte, test_data_byte = shopee_dataset(download_dir, is_bytearray=True)
```

Both image paths and bytearrays are supported for training and inference interchangeably.

## Training

```python
from autogluon.multimodal import MultiModalPredictor
import uuid

model_path = f"./tmp/{uuid.uuid4().hex}-automm_shopee"
predictor = MultiModalPredictor(label="label", path=model_path)
predictor.fit(
    train_data=train_data_path,
    time_limit=30,  # seconds
)
```

**Key parameters**:
- `label`: Target column name
- `path`: Model save directory
- `time_limit`: Training duration control

## Evaluation

```python
scores = predictor.evaluate(test_data_path, metrics=["accuracy"])
print('Top-1 test acc: %.3f' % scores["accuracy"])
```

Models trained on paths work with bytearrays and vice versa.

## Inference

### Predict Class
```python
predictions = predictor.predict({'image': [image_path]})
```

### Predict Probabilities
```python
proba = predictor.predict_proba({'image': [image_path]})
```

### Extract Embeddings
```python
feature = predictor.extract_embedding({'image': [image_path]})
print(feature[0].shape)  # N-dimensional vector (512-2048)
```

All methods support both image paths and bytearrays.

## Save & Load

Model auto-saves after `fit()`.

```python
loaded_predictor = MultiModalPredictor.load(model_path)
```

**⚠️ Security Warning**: `MultiModalPredictor.load()` uses `pickle` internally. **Only load trusted data** - malicious pickle data can execute arbitrary code.

## Additional Resources

- [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm)
- [Customize AutoMM](../advanced_topics/customization.ipynb)