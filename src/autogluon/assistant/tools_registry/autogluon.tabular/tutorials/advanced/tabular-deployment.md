Summary: This tutorial demonstrates AutoGluon model deployment optimization techniques, focusing on efficient model cloning and performance enhancement for production environments. It covers implementation of basic model training, two types of predictor cloning (standard and deployment-optimized), and experimental model compilation for faster inference. Key functionalities include using clone_for_deployment() to minimize artifact size, persist() for memory-based acceleration, and model compilation for specific model types. The tutorial helps with tasks like optimizing model deployment size, improving prediction speed, and managing version compatibility, while highlighting important considerations for storage management and functionality trade-offs in production settings.

# Predicting Columns in a Table - Deployment Optimization

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/autogluon/autogluon/blob/master/docs/tutorials/tabular/advanced/tabular-deployment.ipynb)
[![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/autogluon/autogluon/blob/master/docs/tutorials/tabular/advanced/tabular-deployment.ipynb)



This tutorial will cover how to perform the end-to-end AutoML process to create an optimized and deployable
AutoGluon artifact for production usage.

This tutorial assumes you have already read [Predicting Columns in a Table - Quick Start](../tabular-quick-start.ipynb) and [Predicting Columns in a Table - In Depth](../tabular-indepth.ipynb).

## Fitting a TabularPredictor

We will again use the AdultIncome dataset as in the previous tutorials and train a predictor
to predict whether the person's income exceeds $50,000 or not, which is recorded in the `class` column of this table.


```python
!pip install autogluon.tabular[all]

```


```python
from autogluon.tabular import TabularDataset, TabularPredictor
train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
label = 'class'
subsample_size = 500  # subsample subset of data for faster demo, try setting this to much larger values
train_data = train_data.sample(n=subsample_size, random_state=0)
train_data.head()
```


```python
save_path = 'agModels-predictClass-deployment'  # specifies folder to store trained models
predictor = TabularPredictor(label=label, path=save_path).fit(train_data)
```

Next, load separate test data to demonstrate how to make predictions on new examples at inference time:


```python
test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')
y_test = test_data[label]  # values to predict
test_data.head()
```

We use our trained models to make predictions on the new data:


```python
predictor = TabularPredictor.load(save_path)  # unnecessary, just demonstrates how to load previously-trained predictor from file

y_pred = predictor.predict(test_data)
y_pred
```

We can use leaderboard to evaluate the performance of each individual trained model on our labeled test data:


```python
predictor.leaderboard(test_data)
```

## Snapshot a Predictor with .clone()

Now that we have a working predictor artifact, we may want to alter it in a variety of ways to better suite our needs.
For example, we may want to delete certain models to reduce disk usage via `.delete_models()`,
or train additional models on top of the ones we already have via `.fit_extra()`.

While you can do all of these operations on your predictor,
you may want to be able to be able to revert to a prior state of the predictor in case something goes wrong.
This is where `predictor.clone()` comes in.

`predictor.clone()` allows you to create a snapshot of the given predictor,
cloning the artifacts of the predictor to a new location.
You can then freely play around with the predictor and always load 
the earlier snapshot in case you want to undo your actions.

All you need to do to clone a predictor is specify a new directory path to clone to:


```python
save_path_clone = save_path + '-clone'
# will return the path to the cloned predictor, identical to save_path_clone
path_clone = predictor.clone(path=save_path_clone)
```

Note that this logic doubles disk usage, as it completely clones
every predictor artifact on disk to make an exact replica.

Now we can load the cloned predictor:


```python
predictor_clone = TabularPredictor.load(path=path_clone)
# You can alternatively load the cloned TabularPredictor at the time of cloning:
# predictor_clone = predictor.clone(path=save_path_clone, return_clone=True)
```

We can see that the cloned predictor has the same leaderboard and functionality as the original:


```python
y_pred_clone = predictor.predict(test_data)
y_pred_clone
```


```python
y_pred.equals(y_pred_clone)
```


```python
predictor_clone.leaderboard(test_data)
```

Now let's do some extra logic with the clone, such as calling refit_full:


```python
predictor_clone.refit_full()

predictor_clone.leaderboard(test_data)
```

We can see that we were able to fit additional models, but for whatever reason we may want to undo this operation.

Luckily, our original predictor is untouched!


```python
predictor.leaderboard(test_data)
```

We can simply clone a new predictor from our original, and we will no longer be impacted
by the call to refit_full on the prior clone.

## Snapshot a deployment optimized Predictor via .clone_for_deployment()

Instead of cloning an exact copy, we can instead clone a copy
which has the minimal set of artifacts needed to do prediction.

Note that this optimized clone will have very limited functionality outside of calling predict and predict_proba.
For example, it will be unable to train more models.


```python
save_path_clone_opt = save_path + '-clone-opt'
# will return the path to the cloned predictor, identical to save_path_clone_opt
path_clone_opt = predictor.clone_for_deployment(path=save_path_clone_opt)
```


```python
predictor_clone_opt = TabularPredictor.load(path=path_clone_opt)
```

To avoid loading the model in every prediction call, we can persist the model in memory by:


```python
predictor_clone_opt.persist()
```

We can see that the optimized clone still makes the same predictions:


```python
y_pred_clone_opt = predictor_clone_opt.predict(test_data)
y_pred_clone_opt
```


```python
y_pred.equals(y_pred_clone_opt)
```


```python
predictor_clone_opt.leaderboard(test_data)
```

We can check the disk usage of the optimized clone compared to the original:


```python
size_original = predictor.disk_usage()
size_opt = predictor_clone_opt.disk_usage()
print(f'Size Original:  {size_original} bytes')
print(f'Size Optimized: {size_opt} bytes')
print(f'Optimized predictor achieved a {round((1 - (size_opt/size_original)) * 100, 1)}% reduction in disk usage.')
```

We can also investigate the difference in the files that exist in the original and optimized predictor.

Original:


```python
predictor.disk_usage_per_file()
```

Optimized:


```python
predictor_clone_opt.disk_usage_per_file()
```

## Compile models for maximized inference speed

In order to further improve inference efficiency, we can call `.compile()` to automatically
convert sklearn function calls into their ONNX equivalents.
Note that this is currently an experimental feature, which only improves RandomForest and TabularNeuralNetwork models.
The compilation and inference speed acceleration require installation of `skl2onnx` and `onnxruntime` packages.
To install supported versions of these packages automatically, we can call `pip install autogluon.tabular[skl2onnx]`
on top of an existing AutoGluon installation, or `pip install autogluon.tabular[all,skl2onnx]` on a new AutoGluon installation.

It is important to make sure the predictor is cloned, because once the models are compiled, it won't support fitting.


```python
predictor_clone_opt.compile()
```

With the compiled predictor, the prediction results might not be exactly the same but should be very close.


```python
y_pred_compile_opt = predictor_clone_opt.predict(test_data)
y_pred_compile_opt
```

Now all that is left is to upload the optimized predictor to a centralized storage location such as S3.
To use this predictor in a new machine / system, simply download the artifact to local disk and load the predictor.
Ensure that when loading a predictor you use the same Python version
and AutoGluon version used during training to avoid instability.
