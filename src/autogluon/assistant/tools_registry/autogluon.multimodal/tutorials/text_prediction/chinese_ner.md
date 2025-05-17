Summary: This tutorial demonstrates implementing Chinese Named Entity Recognition using AutoGluon's MultiModalPredictor. It covers essential techniques for setting up NER models with Chinese pretrained transformers (specifically 'hfl/chinese-lert-small'), loading preprocessed datasets, and configuring model training with specific entity labels (brand, product, pattern, misc). The tutorial helps with tasks like training NER models, making predictions, and visualizing results using AutoGluon's built-in functions. Key features include simplified model configuration, integration with Chinese language models, evaluation methods, and visualization tools for NER results, making it valuable for implementing Chinese text entity extraction systems.

# AutoMM for Named Entity Recognition in Chinese - Quick Start

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/autogluon/autogluon/blob/master/docs/tutorials/multimodal/text_prediction/chinese_ner.ipynb)
[![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/autogluon/autogluon/blob/master/docs/tutorials/multimodal/text_prediction/chinese_ner.ipynb)

In this tutorial, we will demonstrate how to use AutoMM for Chinese Named Entity Recognition using an e-commerce dataset extracted from one of the most popular online marketplaces, [TaoBao.com](https://taobao.com). 
The dataset is collected and labelled by [Jie et al.](https://aclanthology.org/N19-1079.pdf) and the text column mainly consists of product descriptions. 
The following figure shows an example of Taobao product description.

![Taobao product description. A rabbit toy for lunar new year decoration.](https://automl-mm-bench.s3.amazonaws.com/ner/images_for_tutorial/chinese_ner.png)


## Load the Data 
We have preprocessed the dataset to make it ready-to-use with AutoMM.


```python
!pip install autogluon.multimodal

```


```python
import autogluon.multimodal
from autogluon.core.utils.loaders import load_pd
from autogluon.multimodal.utils import visualize_ner
train_data = load_pd.load('https://automl-mm-bench.s3.amazonaws.com/ner/taobao-ner/chinese_ner_train.csv')
dev_data = load_pd.load('https://automl-mm-bench.s3.amazonaws.com/ner/taobao-ner/chinese_ner_dev.csv')
train_data.head(5)
```

HPPX, HCCX, XH, and MISC stand for brand, product, pattern, and Miscellaneous information (e.g., product Specification), respectively. 
Let's visualize one of the examples, which is about *online games top up services*.


```python
visualize_ner(train_data["text_snippet"].iloc[0], train_data["entity_annotations"].iloc[0])
```

## Training
With AutoMM, the process of Chinese entity recognition is the same as English entity recognition. 
All you need to do is to select a suitable foundation model checkpoint that are pretrained on Chinese or multilingual documents. 
Here we use the `'hfl/chinese-lert-small'` backbone for demonstration purpose.

Now, let’s create a predictor for named entity recognition by setting the problem_type to ner and specifying the label column. 
Afterwards, we call predictor.fit() to train the model for a few minutes.


```python
from autogluon.multimodal import MultiModalPredictor
import uuid

label_col = "entity_annotations"
model_path = f"./tmp/{uuid.uuid4().hex}-automm_ner"  # You can rename it to the model path you like
predictor = MultiModalPredictor(problem_type="ner", label=label_col, path=model_path)
predictor.fit(
    train_data=train_data,
    hyperparameters={'model.ner_text.checkpoint_name':'hfl/chinese-lert-small'},
    time_limit=300, #second
)
```

## Evaluation 
To check the model performance on the test dataset, all you need to do is to call `predictor.evaluate(...)`.


```python
predictor.evaluate(dev_data)
```

## Prediction and Visualization
You can easily obtain the predictions given an input sentence by by calling `predictor.predict(...)`.


```python
output = predictor.predict(dev_data)
visualize_ner(dev_data["text_snippet"].iloc[0], output[0])
```

Now, let's make predictions on the rabbit toy example.


```python
sentence = "2023年兔年挂件新年装饰品小挂饰乔迁之喜门挂小兔子"
predictions = predictor.predict({'text_snippet': [sentence]})
visualize_ner(sentence, predictions[0])
```

## Other Examples

You may go to [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm) to explore other examples about AutoMM.

## Customization
To learn how to customize AutoMM, please refer to [Customize AutoMM](../advanced_topics/customization.ipynb).
