# Condensed: ```python

Summary: This tutorial demonstrates multilingual text classification with AutoGluon, covering: implementation of sentiment analysis across languages using pre-trained transformer models; techniques for both language-specific model training and zero-shot cross-lingual transfer; and comparison between monolingual and multilingual approaches. Key features include using language-specific BERT models (showing their limitations with other languages), implementing cross-lingual transfer with the "multilingual" preset parameter, and evaluating model performance across English, German, and Japanese datasets without language-specific training. The tutorial helps with building text classifiers that work effectively across multiple languages with minimal language-specific customization.

*This is a condensed version that preserves essential implementation details and context.*

# Multilingual Text Classification with AutoGluon

## Setup and Data Preparation

```python
!pip install autogluon.multimodal
!wget --quiet https://automl-mm-bench.s3.amazonaws.com/multilingual-datasets/amazon_review_sentiment_cross_lingual.zip
!unzip -q -o amazon_review_sentiment_cross_lingual.zip -d .

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Load German data
train_de_df = pd.read_csv('amazon_review_sentiment_cross_lingual/de_train.tsv',
                          sep='\t', header=None, names=['label', 'text']) \
                .sample(1000, random_state=123)
train_de_df.reset_index(inplace=True, drop=True)

test_de_df = pd.read_csv('amazon_review_sentiment_cross_lingual/de_test.tsv',
                          sep='\t', header=None, names=['label', 'text']) \
               .sample(200, random_state=123)
test_de_df.reset_index(inplace=True, drop=True)

# Load English data
train_en_df = pd.read_csv('amazon_review_sentiment_cross_lingual/en_train.tsv',
                          sep='\t', header=None, names=['label', 'text']) \
                .sample(1000, random_state=123)
train_en_df.reset_index(inplace=True, drop=True)

test_en_df = pd.read_csv('amazon_review_sentiment_cross_lingual/en_test.tsv',
                          sep='\t', header=None, names=['label', 'text']) \
               .sample(200, random_state=123)
test_en_df.reset_index(inplace=True, drop=True)
```

## Approach 1: Finetune German BERT

```python
from autogluon.multimodal import MultiModalPredictor

predictor = MultiModalPredictor(label='label')
predictor.fit(train_de_df,
              hyperparameters={
                  'model.hf_text.checkpoint_name': 'bert-base-german-cased',
                  'optim.max_epochs': 2
              })

# Evaluate on German test set
score = predictor.evaluate(test_de_df)
print('Score on the German Testset:', score)

# Evaluate on English test set
score = predictor.evaluate(test_en_df)
print('Score on the English Testset:', score)
```

**Key finding**: The German BERT model performs well on German data but poorly on English data.

## Approach 2: Cross-lingual Transfer

```python
predictor = MultiModalPredictor(label='label')
predictor.fit(train_en_df,
              presets='multilingual',
              hyperparameters={
                  'optim.max_epochs': 2
              })

# Evaluate on English test set
score_in_en = predictor.evaluate(test_en_df)
print('Score in the English Testset:', score_in_en)

# Evaluate on German test set
score_in_de = predictor.evaluate(test_de_df)
print('Score in the German Testset:', score_in_de)

# Test on Japanese data
test_jp_df = pd.read_csv('amazon_review_sentiment_cross_lingual/jp_test.tsv',
                          sep='\t', header=None, names=['label', 'text']) \
               .sample(200, random_state=123)
test_jp_df.reset_index(inplace=True, drop=True)

score_in_jp = predictor.evaluate(test_jp_df)
print('Score in the Japanese Testset:', score_in_jp)
```

**Key implementation details**:
- Use `presets="multilingual"` to enable zero-shot cross-lingual transfer
- AutoGluon automatically uses state-of-the-art models like DeBERTa-V3
- The model trained only on English data works well for German and Japanese without additional training

**Best practice**: For multilingual applications, use the multilingual preset rather than language-specific models when you need to support multiple languages.