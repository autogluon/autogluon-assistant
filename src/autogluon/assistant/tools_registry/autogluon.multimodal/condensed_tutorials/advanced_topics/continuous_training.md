# Condensed: Continuous training provides a method for machine learning models to refine their performance over time. It enables models to build upon previously acquired knowledge, thereby enhancing accuracy, facilitating knowledge transfer across tasks, and saving computational resources. In this tutorial, we will demonstrate three use cases of continuous training with AutoMM.

Summary: This tutorial covers AutoGluon's MultiModalPredictor continuous training capabilities for three scenarios: (1) extending training with additional data/epochs using `MultiModalPredictor.load()` and `.fit()` on new data, (2) resuming interrupted training from checkpoints using `resume=True` parameter, and (3) transfer learning by dumping trained models with `.dump_model()` and loading them via hyperparameters for new tasks. It demonstrates checkpoint management (`model.ckpt`, `last.ckpt`), model evaluation with custom metrics, and supports HuggingFace text, TIMM image, and MMDetection models. Key techniques include incremental training without restarting, fault-tolerant training resumption, and cross-task knowledge transfer through custom checkpoint loading with hyperparameter configuration.

*This is a condensed version that preserves essential implementation details and context.*

# Continuous Training with AutoMM

Continuous training enables models to refine performance over time by building on previously acquired knowledge, enhancing accuracy, facilitating knowledge transfer, and saving computational resources.

## Use Case 1: Extending Training with Additional Data or Epochs

Continue training when models underfit or when new data becomes available. AutoMM supports extending training for the same problem type and classes.

**Example using Stanford Sentiment Treebank (SST) dataset:**

```python
from autogluon.core.utils.loaders import load_pd
from autogluon.multimodal import MultiModalPredictor
import uuid

# Load data
train_data = load_pd.load("https://autogluon-text.s3-accelerate.amazonaws.com/glue/sst/train.parquet")
test_data = load_pd.load("https://autogluon-text.s3-accelerate.amazonaws.com/glue/sst/dev.parquet")
train_data_1 = train_data.sample(n=1000, random_state=0)

# Initial training
model_path = f"./tmp/{uuid.uuid4().hex}-automm_sst"
predictor = MultiModalPredictor(label="label", eval_metric="acc", path=model_path)
predictor.fit(train_data_1, time_limit=60)

# Evaluate
test_score = predictor.evaluate(test_data)
```

**Continue training with additional data:**

```python
# Load checkpoint and continue training
predictor_2 = MultiModalPredictor.load(model_path)
train_data_2 = train_data.drop(train_data_1.index).sample(n=1000, random_state=0)
predictor_2.fit(train_data_2, time_limit=60)
```

**Key point:** Successfully trained models save `model.ckpt` under `model_path`.

## Use Case 2: Resuming from Last Checkpoint

Resume interrupted training using `last.ckpt` saved under `model_path`:

```python
predictor_resume = MultiModalPredictor.load(path=model_path, resume=True)
predictor.fit(train_data, time_limit=60)
```

## Use Case 3: Transfer Learning to New Tasks

Leverage pre-trained models for related tasks by dumping model weights and using them as foundation models.

**Dump trained model:**

```python
dump_model_path = f"./tmp/{uuid.uuid4().hex}-automm_sst"
predictor.dump_model(save_path=dump_model_path)
```

**Apply to new task (e.g., regression on STS Benchmark):**

```python
# Load new dataset
sts_train_data = load_pd.load("https://autogluon-text.s3-accelerate.amazonaws.com/glue/sts/train.parquet")[
    ["sentence1", "sentence2", "score"]
]
sts_test_data = load_pd.load("https://autogluon-text.s3-accelerate.amazonaws.com/glue/sts/dev.parquet")[
    ["sentence1", "sentence2", "score"]
]

# Train with custom model weights
sts_model_path = f"./tmp/{uuid.uuid4().hex}-automm_sts"
predictor_sts = MultiModalPredictor(label="score", path=sts_model_path)
predictor_sts.fit(
    sts_train_data, 
    hyperparameters={"model.hf_text.checkpoint_name": f"{dump_model_path}/hf_text"}, 
    time_limit=30
)

# Evaluate
test_score = predictor_sts.evaluate(sts_test_data, metrics=["rmse", "pearsonr", "spearmanr"])
```

**Supported model types for dumping:**
- HuggingFace text models: `{"model.hf_text.checkpoint_name": dump_model_path}`
- TIMM image models: `{"model.timm_image.checkpoint_name": timm_image_model_path}`
- MMDetection models: `{"model.mmdet_image.checkpoint_name": mmdet_image_model_path}`
- Fusion models comprising the above

**⚠️ Warning:** This approach does not address [Catastrophic Forgetting](https://en.wikipedia.org/wiki/Catastrophic_interference) when transferring to new tasks.

**Best Practice:** For production use, set longer `time_limit` (e.g., 1 hour) or `time_limit=None` instead of the 60-second demo values.