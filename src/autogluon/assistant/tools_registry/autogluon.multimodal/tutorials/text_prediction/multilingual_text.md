Summary: This tutorial demonstrates multilingual text classification with AutoGluon, covering: implementation of sentiment analysis across languages using pre-trained transformer models; techniques for both language-specific model training and zero-shot cross-lingual transfer; and comparison between monolingual and multilingual approaches. Key features include using language-specific BERT models (showing their limitations with other languages), implementing cross-lingual transfer with the "multilingual" preset parameter, and evaluating model performance across English, German, and Japanese datasets without language-specific training. The tutorial helps with building text classifiers that work effectively across multiple languages with minimal language-specific customization.

```python
!pip install autogluon.multimodal

```


```python
!wget --quiet https://automl-mm-bench.s3.amazonaws.com/multilingual-datasets/amazon_review_sentiment_cross_lingual.zip -O amazon_review_sentiment_cross_lingual.zip
!unzip -q -o amazon_review_sentiment_cross_lingual.zip -d .
```


```python
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

train_de_df = pd.read_csv('amazon_review_sentiment_cross_lingual/de_train.tsv',
                          sep='\t', header=None, names=['label', 'text']) \
                .sample(1000, random_state=123)
train_de_df.reset_index(inplace=True, drop=True)

test_de_df = pd.read_csv('amazon_review_sentiment_cross_lingual/de_test.tsv',
                          sep='\t', header=None, names=['label', 'text']) \
               .sample(200, random_state=123)
test_de_df.reset_index(inplace=True, drop=True)
print(train_de_df)
```


```python
train_en_df = pd.read_csv('amazon_review_sentiment_cross_lingual/en_train.tsv',
                          sep='\t',
                          header=None,
                          names=['label', 'text']) \
                .sample(1000, random_state=123)
train_en_df.reset_index(inplace=True, drop=True)

test_en_df = pd.read_csv('amazon_review_sentiment_cross_lingual/en_test.tsv',
                          sep='\t',
                          header=None,
                          names=['label', 'text']) \
               .sample(200, random_state=123)
test_en_df.reset_index(inplace=True, drop=True)
print(train_en_df)
```

## Finetune the German BERT

Our first approach is to finetune the [German BERT model](https://www.deepset.ai/german-bert) pretrained by deepset. 
Since `MultiModalPredictor` integrates with the [Huggingface/Transformers](https://huggingface.co/docs/transformers/index) (as explained in [Customize AutoMM](../advanced_topics/customization.ipynb)), 
we directly load the German BERT model available in Huggingface/Transformers, with the key as [bert-base-german-cased](https://huggingface.co/bert-base-german-cased). 
To simplify the experiment, we also just finetune for 4 epochs.


```python
from autogluon.multimodal import MultiModalPredictor

predictor = MultiModalPredictor(label='label')
predictor.fit(train_de_df,
              hyperparameters={
                  'model.hf_text.checkpoint_name': 'bert-base-german-cased',
                  'optim.max_epochs': 2
              })
```


```python
score = predictor.evaluate(test_de_df)
print('Score on the German Testset:')
print(score)
```


```python
score = predictor.evaluate(test_en_df)
print('Score on the English Testset:')
print(score)
```

We can find that the model can achieve good performance on the German dataset but performs poorly on the English dataset. 
Next, we will show how to enable cross-lingual transfer so you can get a model that can magically work for **both German and English**.

## Cross-lingual Transfer

In the real-world scenario, it is pretty common that you have trained a model for English and would like to extend the model to support other languages like German. 
This setting is also known as cross-lingual transfer. One way to solve the problem is to apply a machine translation model to translate the sentences from the 
other language (e.g., German) to English and apply the English model.
However, as showed in ["Unsupervised Cross-lingual Representation Learning at Scale"](https://arxiv.org/pdf/1911.02116.pdf), 
there is a better and cost-friendlier way for cross lingual transfer, enabled via large-scale multilingual pretraining.
The author showed that via large-scale pretraining, the backbone (called XLM-R) is able to conduct *zero-shot* cross lingual transfer, 
meaning that you can directly apply the model trained in the English dataset to datasets in other languages. 
It also outperforms the baseline "TRANSLATE-TEST", meaning to translate the data from other languages to English and apply the English model. 

In AutoGluon, you can just turn on `presets="multilingual"` in MultiModalPredictor to load a backbone that is suitable for zero-shot transfer. 
Internally, we will automatically use state-of-the-art models like [DeBERTa-V3](https://arxiv.org/abs/2111.09543).


```python
from autogluon.multimodal import MultiModalPredictor

predictor = MultiModalPredictor(label='label')
predictor.fit(train_en_df,
              presets='multilingual',
              hyperparameters={
                  'optim.max_epochs': 2
              })
```


```python
score_in_en = predictor.evaluate(test_en_df)
print('Score in the English Testset:')
print(score_in_en)
```


```python
score_in_de = predictor.evaluate(test_de_df)
print('Score in the German Testset:')
print(score_in_de)
```

We can see that the model works for both German and English!

Let's also inspect the model's performance on Japanese:


```python
test_jp_df = pd.read_csv('amazon_review_sentiment_cross_lingual/jp_test.tsv',
                          sep='\t', header=None, names=['label', 'text']) \
               .sample(200, random_state=123)
test_jp_df.reset_index(inplace=True, drop=True)
print(test_jp_df)
```


```python
print('Negative labe ratio of the Japanese Testset=', test_jp_df['label'].value_counts()[0] / len(test_jp_df))
score_in_jp = predictor.evaluate(test_jp_df)
print('Score in the Japanese Testset:')
print(score_in_jp)
```

Amazingly, the model also works for Japanese!

## Other Examples

You may go to [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm) to explore other examples about AutoMM.

## Customization
To learn how to customize AutoMM, please refer to [Customize AutoMM](../advanced_topics/customization.ipynb).
