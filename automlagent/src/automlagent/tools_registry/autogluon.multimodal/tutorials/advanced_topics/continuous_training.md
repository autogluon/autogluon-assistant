Summary: This tutorial demonstrates continuous training techniques using AutoMM's MultiModalPredictor, covering three main implementation patterns: extending training with new data, resuming interrupted training from checkpoints, and transfer learning with pre-trained models. It helps with tasks involving model persistence, training continuation, and transfer learning for text, image, and fusion models. Key features include checkpoint management (model.ckpt vs last.ckpt), hyperparameter configuration for different model types (HuggingFace, TIMM, MMDetection), and data consistency requirements. The tutorial emphasizes best practices for production deployment and warns about potential catastrophic forgetting in transfer learning scenarios.

# Continuous Training with AutoMM
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/autogluon/autogluon/blob/master/docs/tutorials/multimodal/advanced_topics/continuous_training.ipynb)
[![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/autogluon/autogluon/blob/master/docs/tutorials/multimodal/advanced_topics/continuous_training.ipynb)




Continuous training provides a method for machine learning models to refine their performance over time. It enables models to build upon previously acquired knowledge, thereby enhancing accuracy, facilitating knowledge transfer across tasks, and saving computational resources. In this tutorial, we will demonstrate three use cases of continuous training with AutoMM.

### Use Case 1: Expanding Training with Additional Data or Training Time

Sometimes, the model could benefit from more training epochs or additional training time in case of underfitting. With AutoMM, you can easily extend the training time of your model without starting from scratch.

Additionally, it's also common to need to incorporate more data into your model. AutoMM allows you to continue training with data of the same problem type and same classes if it is a multiclass problem. This flexibility makes it easy to improve and adapt your models as your data grows.

We use [Stanford Sentiment Treebank (SST)](https://nlp.stanford.edu/sentiment/) dataset as an example. It consists of movie reviews and their associated sentiment. Given a new movie review, the goal is to predict the sentiment reflected in the text (in this case a binary classification, where reviews are labeled as 1 if they convey a positive opinion and labeled as 0 otherwise). Let’s first load and look at the data, noting the labels are stored in a column called label.


```python
from autogluon.core.utils.loaders import load_pd

train_data = load_pd.load("https://autogluon-text.s3-accelerate.amazonaws.com/glue/sst/train.parquet")
test_data = load_pd.load("https://autogluon-text.s3-accelerate.amazonaws.com/glue/sst/dev.parquet")
subsample_size = 1000  # subsample data for faster demo, try setting this to larger values
train_data_1 = train_data.sample(n=subsample_size, random_state=0)
train_data_1.head(10)
```

Now let's train the model. To ensure this tutorial runs quickly, we simply call fit() with a subset of 1000 training examples and limit its runtime to approximately 1 minute. To achieve reasonable performance in your applications, you are recommended to set much longer time_limit (eg. 1 hour), or do not specify time_limit at all (time_limit=None).


```python
from autogluon.multimodal import MultiModalPredictor
import uuid

model_path = f"./tmp/{uuid.uuid4().hex}-automm_sst"
predictor = MultiModalPredictor(label="label", eval_metric="acc", path=model_path)
predictor.fit(train_data_1, time_limit=60)
```

After training, we can evaluate our predictor on separate test data formatted similarly to our training data:


```python
test_score = predictor.evaluate(test_data)
print(test_score)
```

If the training was completed successfully, `model.ckpt` can be found under `model_path`. If you think the model still underfits, you can continue training from this checkpoint by just running another `.fit()` with the same data. If you have some new data to add in and don't want to train from scratch, you can also run `.fit()` with the new combined dataset.


```python
predictor_2 = MultiModalPredictor.load(model_path)  # you can also use the `predictor` we assigned above
train_data_2 = train_data.drop(train_data_1.index).sample(n=subsample_size, random_state=0)
predictor_2.fit(train_data_2, time_limit=60)
```


```python
test_score_2 = predictor_2.evaluate(test_data)
print(test_score_2)
```

### Use Case 2: Resuming Training from the Last Checkpoint

If your training process collapsed for some reason, AutoMM allows you to resume training right from where you left off. `last.ckpt` will be saved under `model_path` instead of `model.ckpt`. By resuming the training, you just have to call `MultiModalPredictor.load()` with `resume` option:


```
predictor_resume = MultiModalPredictor.load(path=model_path, resume=True)
predictor.fit(train_data, time_limit=60)
```

### Use Case 3: Applying Pre-Trained Models to New Tasks

Often, you'll encounter situations where a new task is related but not identical to a task you've previously trained a model for (e.g., training a more fine-grained sentiment analysis model, or adding more classes to your multiclass model). If you wish to leverage the knowledge that the model has already learned from the old data to help it learn the new task more quickly and effectively, AutoMM supports dumping your trained models into model weights and using them as foundation models:


```python
dump_model_path = f"./tmp/{uuid.uuid4().hex}-automm_sst"
predictor.dump_model(save_path=dump_model_path)
```

You can then load the weights of the trained model, and continue training / fine-tuning the model on the new data.

Here is an example that uses the binary text model we trained previously on a regression task. We use the [Semantic Textual Similarity Benchmark dataset](https://paperswithcode.com/dataset/sts-benchmark?t) for illustration only, so you might want to apply this feature to more relevant datasets. In this data, the column named score contains numerical values (which we would like to predict) that are human-annotated similarity scores for each given pair of sentences.


```python
sts_train_data = load_pd.load("https://autogluon-text.s3-accelerate.amazonaws.com/glue/sts/train.parquet")[
    ["sentence1", "sentence2", "score"]
]
sts_test_data = load_pd.load("https://autogluon-text.s3-accelerate.amazonaws.com/glue/sts/dev.parquet")[
    ["sentence1", "sentence2", "score"]
]
sts_train_data.head(10)
```

To specify a custom model that you created, use `hyperparameters` option in `.fit()`:

```
hyperparameters={
    "model.hf_text.checkpoint_name": dump_model_path
}
```


```python
sts_model_path = f"./tmp/{uuid.uuid4().hex}-automm_sts"
predictor_sts = MultiModalPredictor(label="score", path=sts_model_path)
predictor_sts.fit(
    sts_train_data, hyperparameters={"model.hf_text.checkpoint_name": f"{dump_model_path}/hf_text"}, time_limit=30
)
```


```python
test_score = predictor_sts.evaluate(sts_test_data, metrics=["rmse", "pearsonr", "spearmanr"])
print("RMSE = {:.2f}".format(test_score["rmse"]))
print("PEARSONR = {:.4f}".format(test_score["pearsonr"]))
print("SPEARMANR = {:.4f}".format(test_score["spearmanr"]))
```

We currently support dumping `timm` image models, `MMDetection` image models, `HuggingFace` text models, and any fusion models that comprises the aforementioned models. Similarly, we can also load a custom trained `timm` image model with:
```
{"model.timm_image.checkpoint_name": timm_image_model_path}
```
and a custom trained `MMDetection` model with:
```
{"model.mmdet_image.checkpoint_name": mmdet_image_model_path}
```

This feature helps you apply the knowledge of your previously trained task onto a new task, which saves your time and computational power. We will not go into details in this tutorial, but do keep in mind that we have not addressed a big challenge in this use case, i.e. [Catastrophic Forgetting](https://en.wikipedia.org/wiki/Catastrophic_interference#:~:text=Catastrophic%20interference%2C%20also%20known%20as,information%20upon%20learning%20new%20information.).
