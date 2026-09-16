# Condensed: ```python

Summary: This tutorial demonstrates parameter-efficient finetuning (PEFT) techniques for multilingual text classification using AutoGluon's MultiModalPredictor. It covers implementing IA3+BitFit optimization (tuning only 0.5% of parameters) for cross-lingual sentiment analysis, and training large models like FLAN-T5-XL (2B parameters) on single GPU using gradient checkpointing. Key implementations include configuring `optim.peft="ia3_bias"` for efficient training, setting hyperparameters for learning rates and batch sizes, enabling `gradient_checkpointing=True` for memory optimization, and evaluating models across multiple languages (English, German, Japanese). The tutorial helps with tasks like memory-efficient finetuning of large language models, multilingual model training with limited resources, and configuring AutoGluon's hyperparameters for optimal performance while achieving comparable results to full finetuning.

*This is a condensed version that preserves essential implementation details and context.*

# Efficient Finetuning for Multilingual Models

## Setup and Data Preparation

```python
!pip install autogluon.multimodal
!wget --quiet https://automl-mm-bench.s3.amazonaws.com/multilingual-datasets/amazon_review_sentiment_cross_lingual.zip -O amazon_review_sentiment_cross_lingual.zip
!unzip -q -o amazon_review_sentiment_cross_lingual.zip -d .
```

**Configure cache and load data:**
```python
import os, shutil, pandas as pd, warnings
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_CACHE"] = "cache"

# Load English training (1000 samples) and test sets (200 samples each)
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

## Finetuning with IA3 + BitFit

**Enable efficient finetuning** by setting `optim.peft` to `"ia3_bias"`:

```python
from autogluon.multimodal import MultiModalPredictor

predictor = MultiModalPredictor(label="label", path=model_path)
predictor.fit(train_en_df,
              presets="multilingual",
              hyperparameters={
                  "optim.peft": "ia3_bias",
                  "optim.lr_decay": 0.9,
                  "optim.lr": 3e-03,
                  "optim.end_lr": 3e-03,
                  "optim.max_epochs": 2,
                  "optim.warmup_steps": 0,
                  "env.batch_size": 32,
              })
```

**Key Benefits:**
- Only **~0.5% of parameters** are tunable
- Achieves **comparable results** to full finetuning
- Model trained on English generalizes well to German/Japanese test sets

```python
score_in_en = predictor.evaluate(test_en_df)
score_in_de = predictor.evaluate(test_de_df)
score_in_jp = predictor.evaluate(test_jp_df)
```

## Training FLAN-T5-XL on Single GPU

**Finetune 2B parameter model** (google/flan-t5-xl) on single T4 GPU using gradient checkpointing + PEFT:

```python
predictor = MultiModalPredictor(label="label", path=new_model_path)
predictor.fit(train_en_df_downsample,  # 200 samples
              presets="multilingual",
              hyperparameters={
                  "model.hf_text.checkpoint_name": "google/flan-t5-xl",
                  "model.hf_text.gradient_checkpointing": True,  # Enable checkpointing
                  "model.hf_text.low_cpu_mem_usage": True,
                  "optim.peft": "ia3_bias",
                  "optim.lr_decay": 0.9,
                  "optim.lr": 3e-03,
                  "optim.end_lr": 3e-03,
                  "optim.max_epochs": 1,
                  "optim.warmup_steps": 0,
                  "env.batch_size": 1,
                  "env.inference_batch_size_ratio": 1
              })
```

**Critical Configurations:**
- `gradient_checkpointing`: Enables training large models with limited memory
- `low_cpu_mem_usage`: Reduces CPU memory footprint
- `batch_size`: 1 for memory constraints
- **Result:** 203K trainable params out of 1.2B total (~0.017%)

**Performance:** Achieved 0.931 ROC-AUC on English test set

## Additional Resources
- [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm)
- [Customize AutoMM](customization.ipynb)