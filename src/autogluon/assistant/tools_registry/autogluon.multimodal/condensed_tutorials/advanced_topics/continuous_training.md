# Condensed: Continuous training provides a method for machine learning models to refine their performance over time. It enables models to build upon previously acquired knowledge, thereby enhancing accuracy, facilitating knowledge transfer across tasks, and saving computational resources. In this tutorial, we will demonstrate three use cases of continuous training with AutoMM.

Summary: This tutorial demonstrates AutoMM's continuous training capabilities with three key use cases: (1) extending model training with additional data or epochs without restarting, (2) resuming training from the last checkpoint after interruptions, and (3) transferring knowledge from pre-trained models to new tasks. It covers implementation techniques for loading/saving models, continuing training with new data, and applying transfer learning across different tasks. The tutorial specifically shows how to reuse weights from text classification for regression tasks and supports transfer learning for HuggingFace text models, TIMM image models, MMDetection models, and fusion models, while warning about potential catastrophic forgetting.

*This is a condensed version that preserves essential implementation details and context.*

# Continuous Training with AutoMM

## Use Case 1: Expanding Training with Additional Data or Training Time

AutoMM allows extending model training without starting from scratch, either by adding more epochs or incorporating new data of the same problem type.

```python
# Load data
from autogluon.core.utils.loaders import load_pd

train_data = load_pd.load("https://autogluon-text.s3-accelerate.amazonaws.com/glue/sst/train.parquet")
test_data = load_pd.load("https://autogluon-text.s3-accelerate.amazonaws.com/glue/sst/dev.parquet")
subsample_size = 1000  # subsample for faster demo
train_data_1 = train_data.sample(n=subsample_size, random_state=0)

# Initial training
from autogluon.multimodal import MultiModalPredictor
import uuid

model_path = f"./tmp/{uuid.uuid4().hex}-automm_sst"
predictor = MultiModalPredictor(label="label", eval_metric="acc", path=model_path)
predictor.fit(train_data_1, time_limit=60)

# Evaluate
test_score = predictor.evaluate(test_data)
print(test_score)

# Continue training with new data
predictor_2 = MultiModalPredictor.load(model_path)
train_data_2 = train_data.drop(train_data_1.index).sample(n=subsample_size, random_state=0)
predictor_2.fit(train_data_2, time_limit=60)

test_score_2 = predictor_2.evaluate(test_data)
print(test_score_2)
```

## Use Case 2: Resuming Training from the Last Checkpoint

If training collapses, resume from the last checkpoint:

```python
predictor_resume = MultiModalPredictor.load(path=model_path, resume=True)
predictor.fit(train_data, time_limit=60)
```

## Use Case 3: Applying Pre-Trained Models to New Tasks

Transfer knowledge from a trained model to a related but different task:

```python
# Dump model weights
dump_model_path = f"./tmp/{uuid.uuid4().hex}-automm_sst"
predictor.dump_model(save_path=dump_model_path)

# Load regression dataset
sts_train_data = load_pd.load("https://autogluon-text.s3-accelerate.amazonaws.com/glue/sts/train.parquet")[
    ["sentence1", "sentence2", "score"]
]
sts_test_data = load_pd.load("https://autogluon-text.s3-accelerate.amazonaws.com/glue/sts/dev.parquet")[
    ["sentence1", "sentence2", "score"]
]

# Train on new task using previous model weights
sts_model_path = f"./tmp/{uuid.uuid4().hex}-automm_sts"
predictor_sts = MultiModalPredictor(label="score", path=sts_model_path)
predictor_sts.fit(
    sts_train_data, 
    hyperparameters={"model.hf_text.checkpoint_name": f"{dump_model_path}/hf_text"}, 
    time_limit=30
)

# Evaluate
test_score = predictor_sts.evaluate(sts_test_data, metrics=["rmse", "pearsonr", "spearmanr"])
print("RMSE = {:.2f}".format(test_score["rmse"]))
print("PEARSONR = {:.4f}".format(test_score["pearsonr"]))
print("SPEARMANR = {:.4f}".format(test_score["spearmanr"]))
```

### Supported Model Types for Transfer Learning

- HuggingFace text models: `{"model.hf_text.checkpoint_name": hf_text_model_path}`
- TIMM image models: `{"model.timm_image.checkpoint_name": timm_image_model_path}`
- MMDetection models: `{"model.mmdet_image.checkpoint_name": mmdet_image_model_path}`
- Any fusion models comprising the above models

**Note:** Be aware of potential catastrophic forgetting when applying pre-trained models to new tasks.