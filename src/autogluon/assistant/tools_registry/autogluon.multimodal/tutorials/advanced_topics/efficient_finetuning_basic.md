Summary: This tutorial demonstrates parameter-efficient multilingual model fine-tuning using AutoGluon's MultiModalPredictor. It covers implementing IA3+BitFit techniques that require only ~0.5% of parameters while maintaining cross-lingual performance across English, German, and Japanese sentiment analysis. The tutorial shows how to train large language models (like FLAN-T5-XL) on limited hardware using gradient checkpointing, and provides code for data preparation, model configuration, and evaluation. Key functionalities include PEFT implementation, multilingual transfer learning, memory optimization techniques, and hyperparameter configuration for efficient fine-tuning of transformer models.

```python
!pip install autogluon.multimodal

```


```python
!wget --quiet https://automl-mm-bench.s3.amazonaws.com/multilingual-datasets/amazon_review_sentiment_cross_lingual.zip -O amazon_review_sentiment_cross_lingual.zip
!unzip -q -o amazon_review_sentiment_cross_lingual.zip -d .
```


```python
import os
import shutil
os.environ["TRANSFORMERS_CACHE"] = "cache"

def clear_cache():
    if os.path.exists("cache"):
        shutil.rmtree("cache")

clear_cache()
```


```python
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

train_en_df = pd.read_csv("amazon_review_sentiment_cross_lingual/en_train.tsv",
                          sep="\t",
                          header=None,
                          names=["label", "text"]) \
                .sample(1000, random_state=123).reset_index(drop=True)

test_en_df = pd.read_csv("amazon_review_sentiment_cross_lingual/en_test.tsv",
                          sep="\t",
                          header=None,
                          names=["label", "text"]) \
               .sample(200, random_state=123).reset_index(drop=True)
test_de_df = pd.read_csv("amazon_review_sentiment_cross_lingual/de_test.tsv",
                          sep="\t", header=None, names=["label", "text"]) \
               .sample(200, random_state=123).reset_index(drop=True)

test_jp_df = pd.read_csv('amazon_review_sentiment_cross_lingual/jp_test.tsv',
                          sep='\t', header=None, names=['label', 'text']) \
               .sample(200, random_state=123).reset_index(drop=True)
train_en_df.head(5)
```


```python
test_jp_df.head(5)
```

## Finetuning Multilingual Model with IA3 + BitFit

In AutoMM, to enable efficient finetuning, just specify the `optim.peft` to be `"ia3_bias"`.


```python
from autogluon.multimodal import MultiModalPredictor
import uuid

model_path = f"./tmp/{uuid.uuid4().hex}-multilingual_ia3"
predictor = MultiModalPredictor(label="label",
                                path=model_path)
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

The fraction of the tunable parameters is around **0.5%** of all parameters. Actually, the model trained purely on English data can achieve good performance 
on the test sets, even on the German / Japanese test set. It obtained **comparable results** as full-finetuning as in [AutoMM for Text - Multilingual Problems](../text_prediction/multilingual_text.ipynb).


```python
score_in_en = predictor.evaluate(test_en_df)
score_in_de = predictor.evaluate(test_de_df)
score_in_jp = predictor.evaluate(test_jp_df)
print('Score in the English Testset:', score_in_en)
print('Score in the German Testset:', score_in_de)
print('Score in the Japanese Testset:', score_in_jp)
```

## Training FLAN-T5-XL on Single GPU

By combining [gradient checkpointing](https://pytorch.org/docs/stable/checkpoint.html) and parameter-efficient finetuning, it is feasible to finetune 
[google/flan-t5-xl](https://huggingface.co/google/flan-t5-xl) that has close to two billion parameterswith a single T4 GPU available in
[AWS G4 instances](https://aws.amazon.com/ec2/instance-types/g4/). 
To turn on gradient checkpointing, you just need to set `"model.hf_text.gradient_checkpointing"` to `True`. 
To accelerate the training, we downsample the number of training samples to be 200.


```python
# Just for clean the space
clear_cache()
shutil.rmtree(model_path)
```

```python
from autogluon.multimodal import MultiModalPredictor

train_en_df_downsample = train_en_df.sample(200, random_state=123)

new_model_path = f"./tmp/{uuid.uuid4().hex}-multilingual_ia3_gradient_checkpoint"
predictor = MultiModalPredictor(label="label",
                                path=new_model_path)
predictor.fit(train_en_df_downsample,
              presets="multilingual",
              hyperparameters={
                  "model.hf_text.checkpoint_name": "google/flan-t5-xl",
                  "model.hf_text.gradient_checkpointing": True,
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


```
Global seed set to 123
Auto select gpus: [0]
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
HPU available: False, using: 0 HPUs
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

  | Name              | Type                         | Params
-------------------------------------------------------------------
0 | model             | HFAutoModelForTextPrediction | 1.2 B 
1 | validation_metric | AUROC                        | 0     
2 | loss_func         | CrossEntropyLoss             | 0     
-------------------------------------------------------------------
203 K     Trainable params
1.2 B     Non-trainable params
1.2 B     Total params
4,894.913 Total estimated model params size (MB)
Epoch 0, global step 20: 'val_roc_auc' reached 0.88802 (best 0.88802), saving model to '/home/ubuntu/autogluon/docs/tutorials/multimodal/advanced_topics/multilingual_ia3_gradient_checkpoint/epoch=0-step=20.ckpt' as top 1
Epoch 0, global step 40: 'val_roc_auc' reached 0.94531 (best 0.94531), saving model to '/home/ubuntu/autogluon/docs/tutorials/multimodal/advanced_topics/multilingual_ia3_gradient_checkpoint/epoch=0-step=40.ckpt' as top 1
`Trainer.fit` stopped: `max_epochs=1` reached.





<autogluon.multimodal.predictor.MultiModalPredictor at 0x7fd58c4dbca0>
```


```python
score_in_en = predictor.evaluate(test_en_df)
print('Score in the English Testset:', score_in_en)
```


```
Score in the English Testset: {'roc_auc': 0.931263189629183}
```


```python
# Just for clean the space
clear_cache()
shutil.rmtree(new_model_path)
```


## Other Examples

You may go to [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm) to explore other examples about AutoMM.

## Customization
To learn how to customize AutoMM, please refer to [Customize AutoMM](customization.ipynb).
