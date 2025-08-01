# Condensed: ```python

Summary: This tutorial demonstrates parameter-efficient multilingual model fine-tuning using AutoGluon's MultiModalPredictor. It covers implementing IA3+BitFit techniques that require only ~0.5% of parameters while maintaining cross-lingual performance across English, German, and Japanese sentiment analysis. The tutorial shows how to train large language models (like FLAN-T5-XL) on limited hardware using gradient checkpointing, and provides code for data preparation, model configuration, and evaluation. Key functionalities include PEFT implementation, multilingual transfer learning, memory optimization techniques, and hyperparameter configuration for efficient fine-tuning of transformer models.

*This is a condensed version that preserves essential implementation details and context.*

# Efficient Multilingual Model Fine-tuning with AutoGluon

## Setup and Data Preparation

```python
!pip install autogluon.multimodal
!wget --quiet https://automl-mm-bench.s3.amazonaws.com/multilingual-datasets/amazon_review_sentiment_cross_lingual.zip
!unzip -q -o amazon_review_sentiment_cross_lingual.zip -d .

import os
import shutil
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Set cache directory and clear it
os.environ["TRANSFORMERS_CACHE"] = "cache"
def clear_cache():
    if os.path.exists("cache"):
        shutil.rmtree("cache")
clear_cache()

# Load and sample data
train_en_df = pd.read_csv("amazon_review_sentiment_cross_lingual/en_train.tsv",
                          sep="\t", header=None, names=["label", "text"]) \
                .sample(1000, random_state=123).reset_index(drop=True)

test_en_df = pd.read_csv("amazon_review_sentiment_cross_lingual/en_test.tsv",
                          sep="\t", header=None, names=["label", "text"]) \
               .sample(200, random_state=123).reset_index(drop=True)
               
test_de_df = pd.read_csv("amazon_review_sentiment_cross_lingual/de_test.tsv",
                          sep="\t", header=None, names=["label", "text"]) \
               .sample(200, random_state=123).reset_index(drop=True)
               
test_jp_df = pd.read_csv('amazon_review_sentiment_cross_lingual/jp_test.tsv',
                          sep='\t', header=None, names=['label', 'text']) \
               .sample(200, random_state=123).reset_index(drop=True)
```

## Finetuning Multilingual Model with IA3 + BitFit

Parameter-efficient fine-tuning using IA3 + BitFit requires only ~0.5% of parameters:

```python
from autogluon.multimodal import MultiModalPredictor
import uuid

model_path = f"./tmp/{uuid.uuid4().hex}-multilingual_ia3"
predictor = MultiModalPredictor(label="label", path=model_path)
predictor.fit(
    train_en_df,
    presets="multilingual",
    hyperparameters={
        "optim.peft": "ia3_bias",  # Enable IA3 + BitFit
        "optim.lr_decay": 0.9,
        "optim.lr": 3e-03,
        "optim.end_lr": 3e-03,
        "optim.max_epochs": 2,
        "optim.warmup_steps": 0,
        "env.batch_size": 32,
    }
)

# Evaluate on multiple languages
score_in_en = predictor.evaluate(test_en_df)
score_in_de = predictor.evaluate(test_de_df)
score_in_jp = predictor.evaluate(test_jp_df)
print('Score in the English Testset:', score_in_en)
print('Score in the German Testset:', score_in_de)
print('Score in the Japanese Testset:', score_in_jp)
```

## Training FLAN-T5-XL on Single GPU

Combining gradient checkpointing with parameter-efficient fine-tuning enables training large models on limited hardware:

```python
clear_cache()
shutil.rmtree(model_path)

train_en_df_downsample = train_en_df.sample(200, random_state=123)

new_model_path = f"./tmp/{uuid.uuid4().hex}-multilingual_ia3_gradient_checkpoint"
predictor = MultiModalPredictor(label="label", path=new_model_path)
predictor.fit(
    train_en_df_downsample,
    presets="multilingual",
    hyperparameters={
        "model.hf_text.checkpoint_name": "google/flan-t5-xl",
        "model.hf_text.gradient_checkpointing": True,  # Enable gradient checkpointing
        "model.hf_text.low_cpu_mem_usage": True,
        "optim.peft": "ia3_bias",
        "optim.lr_decay": 0.9,
        "optim.lr": 3e-03,
        "optim.end_lr": 3e-03,
        "optim.max_epochs": 1,
        "optim.warmup_steps": 0,
        "env.batch_size": 1,
        "env.inference_batch_size_ratio": 1
    }
)

score_in_en = predictor.evaluate(test_en_df)
print('Score in the English Testset:', score_in_en)
```

## Key Takeaways

1. Parameter-efficient fine-tuning with `optim.peft="ia3_bias"` requires only ~0.5% of parameters while maintaining performance
2. Models trained on English data can perform well on other languages (German, Japanese)
3. Gradient checkpointing (`model.hf_text.gradient_checkpointing=True`) enables training large models like FLAN-T5-XL (2B parameters) on a single GPU
4. For large models, use smaller batch sizes and enable `low_cpu_mem_usage`

For more examples, see [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm) and [Customize AutoMM](customization.ipynb).