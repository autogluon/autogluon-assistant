Summary: This tutorial demonstrates PDF document classification using AutoGluon's MultiModalPredictor, specifically implementing document processing and classification tasks. It covers essential techniques for handling PDF documents, including data preparation, model training with LayoutLM, and extraction of document embeddings. Key functionalities include automatic PDF processing, text recognition, probability predictions, and embedding extraction. The tutorial helps with tasks like setting up document paths, training classifiers, making predictions, and evaluating model performance. It provides implementation details for configuring the document transformer, managing training time limits, and handling PDF datasets effectively using AutoGluon's multimodal capabilities.

# Classifying PDF Documents with AutoMM

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/autogluon/autogluon/blob/master/docs/tutorials/multimodal/document_prediction/pdf_classification.ipynb)
[![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/autogluon/autogluon/blob/master/docs/tutorials/multimodal/document_prediction/pdf_classification.ipynb)

PDF comes short from Portable Document Format and is one of the most popular document formats.
We can find PDFs everywhere, from personal resumes to business contracts, and from commercial brochures to government documents. 
The list can be endless. 
PDF is highly praised for its portability. 
There's no worry about the receiver being unable to view the document or see an imperfect version regardless of their operating system and device models.

Using AutoMM, you can handle and build machine learning models on PDF documents just like working on other modalities such as text and images, without bothering about PDFs processing. 
In this tutorial, we will introduce how to classify PDF documents automatically with AutoMM using document foundation models. Let’s get started!

For document processing, AutoGluon requires poppler to be installed. Check https://poppler.freedesktop.org for source 

https://github.com/oschwartz10612/poppler-windows for Windows release (make sure to add the bin/ folder to PATH after installing) 

`brew install poppler` for Mac

## Get the PDF document dataset
We have created a simple PDFs dataset via manual crawling for demonstration purpose. 
It consists of two categories, resume and historical documents (downloaded from [milestone documents](https://www.archives.gov/milestone-documents/list)). 
We picked 20 PDF documents for each of the category. 

Now, let's download the dataset and split it into training and test sets.


```python
!pip install autogluon.multimodal

```


```python
import warnings
warnings.filterwarnings('ignore')
import os
import pandas as pd
from autogluon.core.utils.loaders import load_zip

download_dir = './ag_automm_tutorial_pdf_classifier'
zip_file = "https://automl-mm-bench.s3.amazonaws.com/doc_classification/pdf_docs_small.zip"
load_zip.unzip(zip_file, unzip_dir=download_dir)

dataset_path = os.path.join(download_dir, "pdf_docs_small")
pdf_docs = pd.read_csv(f"{dataset_path}/data.csv")
train_data = pdf_docs.sample(frac=0.8, random_state=200)
test_data = pdf_docs.drop(train_data.index)
```

Now, let's visualize one of the PDF documents. Here, we use the S3 URL of the PDF document and `IFrame` to show it in the tutorial.


```python
from IPython.display import IFrame
IFrame("https://automl-mm-bench.s3.amazonaws.com/doc_classification/historical_1.pdf", width=400, height=500)
```

As you can see, this document is an America's historical document in PDF format. 
To make sure the MultiModalPredictor can locate the documents correctly, we need to overwrite the document paths.


```python
from autogluon.multimodal.utils.misc import path_expander

DOC_PATH_COL = "doc_path"

train_data[DOC_PATH_COL] = train_data[DOC_PATH_COL].apply(lambda ele: path_expander(ele, base_folder=download_dir))
test_data[DOC_PATH_COL] = test_data[DOC_PATH_COL].apply(lambda ele: path_expander(ele, base_folder=download_dir))
print(test_data.head())
```

## Create a PDF Document Classifier

You can create a PDFs classifier easily with `MultiModalPredictor`. 
All you need to do is to create a predictor and fit it with the above training dataset. 
AutoMM will handle all the details, like (1) detecting if it is PDF format datasets; (2) processing PDFs like converting it into a format that our model can recognize; (3) detecting and recognizing the text in PDF documents; etc., without your notice. 

Here, label is the name of the column that contains the target variable to predict, e.g., it is “label” in our example. 
We set the training time limit to 120 seconds for demonstration purposes.


```python
from autogluon.multimodal import MultiModalPredictor

predictor = MultiModalPredictor(label="label")
predictor.fit(
    train_data=train_data,
    hyperparameters={"model.document_transformer.checkpoint_name":"microsoft/layoutlm-base-uncased",
    "optimization.top_k_average_method":"best",
    },
    time_limit=120,
)
```

## Evaluate on Test Dataset

You can evaluate the classifier on the test dataset to see how it performs:


```python
scores = predictor.evaluate(test_data, metrics=["accuracy"])
print('The test acc: %.3f' % scores["accuracy"])
```

## Predict on a New PDF Document

Given an example PDF document, we can easily use the final model to predict the label:



```python
predictions = predictor.predict({DOC_PATH_COL: [test_data.iloc[0][DOC_PATH_COL]]})
print(f"Ground-truth label: {test_data.iloc[0]['label']}, Prediction: {predictions}")

```

If probabilities of all categories are needed, you can call predict_proba:


```python
proba = predictor.predict_proba({DOC_PATH_COL: [test_data.iloc[0][DOC_PATH_COL]]})
print(proba)
```

## Extract Embeddings

Extracting representation from the whole document learned by a model is also very useful. 
We provide extract_embedding function to allow predictor to return the N-dimensional document feature where N depends on the model.


```python
feature = predictor.extract_embedding({DOC_PATH_COL: [test_data.iloc[0][DOC_PATH_COL]]})
print(feature[0].shape)
```

## Other Examples

You may go to [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm) to explore other examples about AutoMM.

## Customization
To learn how to customize AutoMM, please refer to [Customize AutoMM](../advanced_topics/customization.ipynb).

