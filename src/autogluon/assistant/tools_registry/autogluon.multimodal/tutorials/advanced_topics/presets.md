Summary: This tutorial demonstrates AutoGluon's MultiModalPredictor for text classification using different quality presets. It covers implementation techniques for sentiment analysis with three preset configurations: medium_quality (fast, smaller models), high_quality (balanced), and best_quality (performance-focused). The tutorial shows how to add hyperparameter optimization with _hpo suffixes, evaluate models using metrics like roc_auc, and view preset configurations. Key functionalities include automatic model selection, time-limited training, and hyperparameter tuning for text classification tasks with minimal code, making it useful for quickly implementing sentiment analysis or other text classification problems.

```python
!pip install autogluon.multimodal

```


```python
import warnings

warnings.filterwarnings('ignore')
```

## Dataset

For demonstration, we use a subsampled Stanford Sentiment Treebank ([SST](https://nlp.stanford.edu/sentiment/)) dataset, which consists of movie reviews and their associated sentiment. 
Given a new movie review, the goal is to predict the sentiment reflected in the text (in this case, a **binary classification**, where reviews are 
labeled as 1 if they conveyed a positive opinion and 0 otherwise).
To get started, let's download and prepare the dataset.


```python
from autogluon.core.utils.loaders import load_pd

train_data = load_pd.load('https://autogluon-text.s3-accelerate.amazonaws.com/glue/sst/train.parquet')
test_data = load_pd.load('https://autogluon-text.s3-accelerate.amazonaws.com/glue/sst/dev.parquet')
subsample_size = 1000  # subsample data for faster demo, try setting this to larger values
train_data = train_data.sample(n=subsample_size, random_state=0)
train_data.head(10)
```

## Medium Quality
In some situations, we prefer fast training and inference to the prediction quality. `medium_quality` is designed for this purpose.
Among the three presets, `medium_quality` has the smallest model size. Now let's fit the predictor using the `medium_quality` preset. Here we set a tight time budget for a quick demo.


```python
from autogluon.multimodal import MultiModalPredictor

predictor = MultiModalPredictor(label='label', eval_metric='acc', presets="medium_quality")
predictor.fit(
    train_data=train_data,
    time_limit=20, # seconds
)
```

Then we can evaluate the predictor on the test data.


```python
scores = predictor.evaluate(test_data, metrics=["roc_auc"])
scores
```

## High Quality
If you want to balance the prediction quality and training/inference speed, you can try the `high_quality` preset, which uses a larger model than `medium_quality`. Accordingly, we need to increase the time limit since larger models require more time to train.


```python
from autogluon.multimodal import MultiModalPredictor

predictor = MultiModalPredictor(label='label', eval_metric='acc', presets="high_quality")
predictor.fit(
    train_data=train_data,
    time_limit=20, # seconds
)
```

Although `high_quality` requires more training time than `medium_quality`, it also brings performance gains.


```python
scores = predictor.evaluate(test_data, metrics=["roc_auc"])
scores
```

## Best Quality
If you want the best performance and don't care about the training/inference cost, give it a try for the `best_quality` preset. High-end GPUs with large memory are preferred in this case. Compared to `high_quality`, it requires much longer training time.


```python
from autogluon.multimodal import MultiModalPredictor

predictor = MultiModalPredictor(label='label', eval_metric='acc', presets="best_quality")
predictor.fit(train_data=train_data, time_limit=180)
```

We can see that `best_quality` achieves better performance than `high_quality`.


```python
scores = predictor.evaluate(test_data, metrics=["roc_auc"])
scores
```

## HPO Presets
The above three presets all use the default hyperparameters, which might not be optimal for your tasks. Fortunately, we also support hyperparameter optimization (HPO) with simple presets. To perform HPO, you can add a postfix `_hpo` in the three presets, resulting in `medium_quality_hpo`, `high_quality_hpo`, and `best_quality_hpo`.

## Display Presets
In case you want to see each preset's inside details, we provide you with a util function to get the hyperparameter setups. For example, here are hyperparameters of preset `high_quality`.


```python
import json
from autogluon.multimodal.utils.presets import get_presets

hyperparameters, hyperparameter_tune_kwargs = get_presets(problem_type="default", presets="high_quality")
print(f"hyperparameters: {json.dumps(hyperparameters, sort_keys=True, indent=4)}")
print(f"hyperparameter_tune_kwargs: {json.dumps(hyperparameter_tune_kwargs, sort_keys=True, indent=4)}")
```

The HPO presets make several hyperparameters tunable such as model backbone, batch size, learning rate, max epoch, and optimizer type. Below are the details of preset `high_quality_hpo`.


```python
import json
import yaml
from autogluon.multimodal.utils.presets import get_presets

hyperparameters, hyperparameter_tune_kwargs = get_presets(problem_type="default", presets="high_quality_hpo")
print(f"hyperparameters: {yaml.dump(hyperparameters, allow_unicode=True, default_flow_style=False)}")
print(f"hyperparameter_tune_kwargs: {json.dumps(hyperparameter_tune_kwargs, sort_keys=True, indent=4)}")
```

## Other Examples
You may go to [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm) to explore other examples about AutoMM.

## Customization
To learn how to customize AutoMM, please refer to [Customize AutoMM](../advanced_topics/customization.ipynb).
