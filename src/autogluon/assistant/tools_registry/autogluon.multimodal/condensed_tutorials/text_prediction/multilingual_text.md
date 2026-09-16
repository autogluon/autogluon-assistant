# Condensed: ```python

Summary: This tutorial demonstrates multilingual text classification using AutoGluon's MultiModalPredictor with two approaches: (1) language-specific finetuning using Huggingface models via `model.hf_text.checkpoint_name` parameter, and (2) zero-shot cross-lingual transfer using `presets='multilingual'` which leverages DeBERTa-V3/XLM-R models. It covers loading multilingual datasets (German, English, Japanese Amazon reviews), training sentiment classifiers, and evaluating cross-lingual performance. Key implementation details include setting hyperparameters like `optim.max_epochs`, using `predictor.fit()` and `predictor.evaluate()` methods, and integrating custom Huggingface transformers. The tutorial helps with building multilingual NLP classifiers that can train on one language and generalize to others without translation, making it useful for cross-lingual transfer learning tasks, sentiment analysis across languages, and cost-effective multilingual model deployment.

*This is a condensed version that preserves essential implementation details and context.*

# Multilingual Text Classification with AutoGluon

## Setup and Data Loading

```python
!pip install autogluon.multimodal

# Download multilingual Amazon review dataset
!wget --quiet https://automl-mm-bench.s3.amazonaws.com/multilingual-datasets/amazon_review_sentiment_cross_lingual.zip -O amazon_review_sentiment_cross_lingual.zip
!unzip -q -o amazon_review_sentiment_cross_lingual.zip -d .
```

```python
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Load German data (1000 train, 200 test samples)
train_de_df = pd.read_csv('amazon_review_sentiment_cross_lingual/de_train.tsv',
                          sep='\t', header=None, names=['label', 'text']) \
                .sample(1000, random_state=123)

test_de_df = pd.read_csv('amazon_review_sentiment_cross_lingual/de_test.tsv',
                          sep='\t', header=None, names=['label', 'text']) \
               .sample(200, random_state=123)

# Load English data
train_en_df = pd.read_csv('amazon_review_sentiment_cross_lingual/en_train.tsv',
                          sep='\t', header=None, names=['label', 'text']) \
                .sample(1000, random_state=123)

test_en_df = pd.read_csv('amazon_review_sentiment_cross_lingual/en_test.tsv',
                          sep='\t', header=None, names=['label', 'text']) \
               .sample(200, random_state=123)
```

## Approach 1: Language-Specific Model (German BERT)

Finetune [German BERT](https://huggingface.co/bert-base-german-cased) using Huggingface integration:

```python
from autogluon.multimodal import MultiModalPredictor

predictor = MultiModalPredictor(label='label')
predictor.fit(train_de_df,
              hyperparameters={
                  'model.hf_text.checkpoint_name': 'bert-base-german-cased',
                  'optim.max_epochs': 2
              })

# Good on German, poor on English
score = predictor.evaluate(test_de_df)
score = predictor.evaluate(test_en_df)
```

**Limitation**: Language-specific models perform poorly on other languages.

## Approach 2: Cross-Lingual Transfer (Recommended)

Use `presets='multilingual'` for zero-shot cross-lingual transfer. Internally uses state-of-the-art models like [DeBERTa-V3](https://arxiv.org/abs/2111.09543) and [XLM-R](https://arxiv.org/pdf/1911.02116.pdf).

```python
predictor = MultiModalPredictor(label='label')
predictor.fit(train_en_df,
              presets='multilingual',  # Key parameter for multilingual support
              hyperparameters={
                  'optim.max_epochs': 2
              })

# Evaluate on English (training language)
score_in_en = predictor.evaluate(test_en_df)

# Zero-shot transfer to German
score_in_de = predictor.evaluate(test_de_df)

# Zero-shot transfer to Japanese
test_jp_df = pd.read_csv('amazon_review_sentiment_cross_lingual/jp_test.tsv',
                          sep='\t', header=None, names=['label', 'text']) \
               .sample(200, random_state=123)
score_in_jp = predictor.evaluate(test_jp_df)
```

**Key Benefits**:
- Train on one language (e.g., English), works on multiple languages
- Outperforms translate-then-predict approaches
- No translation model required
- Cost-effective solution

**Best Practice**: Use `presets='multilingual'` when you need cross-lingual capabilities, even if training on a single language.

## Additional Resources
- [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm)
- [Customize AutoMM](../advanced_topics/customization.ipynb)