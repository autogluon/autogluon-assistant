# Condensed: ```python

Summary: This tutorial demonstrates AutoGluon's MultilabelPredictor implementation for predicting multiple target columns simultaneously. It covers creating a custom class that manages multiple TabularPredictors, with methods for training, prediction, evaluation, and model persistence. Key features include handling different problem types (regression, classification) for each target, optional correlation consideration between labels during prediction, and customizable evaluation metrics. The code helps with multi-target machine learning tasks, offering flexibility in how predictions are generated and evaluated. The implementation provides a clean API for training multiple models with a single interface while maintaining individual model access.

*This is a condensed version that preserves essential implementation details and context.*

# MultilabelPredictor for AutoGluon

## Installation
```python
!pip install autogluon.tabular[all]
```

## Implementation

```python
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.common.utils.utils import setup_outputdir
from autogluon.core.utils.loaders import load_pkl
from autogluon.core.utils.savers import save_pkl
import os.path

class MultilabelPredictor:
    """Tabular Predictor for predicting multiple columns in table."""
    
    multi_predictor_file = 'multilabel_predictor.pkl'
    
    def __init__(self, labels, path=None, problem_types=None, eval_metrics=None, consider_labels_correlation=True, **kwargs):
        if len(labels) < 2:
            raise ValueError("MultilabelPredictor is only intended for predicting MULTIPLE labels")
        if (problem_types is not None) and (len(problem_types) != len(labels)):
            raise ValueError("If provided, `problem_types` must have same length as `labels`")
        if (eval_metrics is not None) and (len(eval_metrics) != len(labels)):
            raise ValueError("If provided, `eval_metrics` must have same length as `labels`")
            
        self.path = setup_outputdir(path, warn_if_exist=False)
        self.labels = labels
        self.consider_labels_correlation = consider_labels_correlation
        self.predictors = {}
        self.eval_metrics = {} if eval_metrics is None else {labels[i]: eval_metrics[i] for i in range(len(labels))}
        
        for i in range(len(labels)):
            label = labels[i]
            path_i = os.path.join(self.path, "Predictor_" + str(label))
            problem_type = problem_types[i] if problem_types is not None else None
            eval_metric = eval_metrics[i] if eval_metrics is not None else None
            self.predictors[label] = TabularPredictor(label=label, problem_type=problem_type, 
                                                     eval_metric=eval_metric, path=path_i, **kwargs)
```

### Key Methods

```python
def fit(self, train_data, tuning_data=None, **kwargs):
    """Fits a separate TabularPredictor to predict each label"""
    if isinstance(train_data, str):
        train_data = TabularDataset(train_data)
    if tuning_data is not None and isinstance(tuning_data, str):
        tuning_data = TabularDataset(tuning_data)
    
    train_data_og = train_data.copy()
    tuning_data_og = tuning_data.copy() if tuning_data is not None else None
    save_metrics = len(self.eval_metrics) == 0
    
    for i in range(len(self.labels)):
        label = self.labels[i]
        predictor = self.get_predictor(label)
        
        # Determine which labels to drop based on correlation setting
        if not self.consider_labels_correlation:
            labels_to_drop = [l for l in self.labels if l != label]
        else:
            labels_to_drop = [self.labels[j] for j in range(i+1, len(self.labels))]
            
        train_data = train_data_og.drop(labels_to_drop, axis=1)
        if tuning_data is not None:
            tuning_data = tuning_data_og.drop(labels_to_drop, axis=1)
            
        print(f"Fitting TabularPredictor for label: {label} ...")
        predictor.fit(train_data=train_data, tuning_data=tuning_data, **kwargs)
        self.predictors[label] = predictor.path
        
        if save_metrics:
            self.eval_metrics[label] = predictor.eval_metric
    self.save()

def predict(self, data, **kwargs):
    """Returns DataFrame with predictions for each label"""
    return self._predict(data, as_proba=False, **kwargs)

def predict_proba(self, data, **kwargs):
    """Returns dict with probability predictions for each label"""
    return self._predict(data, as_proba=True, **kwargs)

def evaluate(self, data, **kwargs):
    """Evaluates predictions for all labels"""
    data = self._get_data(data)
    eval_dict = {}
    for label in self.labels:
        print(f"Evaluating TabularPredictor for label: {label} ...")
        predictor = self.get_predictor(label)
        eval_dict[label] = predictor.evaluate(data, **kwargs)
        if self.consider_labels_correlation:
            data[label] = predictor.predict(data, **kwargs)
    return eval_dict

def save(self):
    """Save MultilabelPredictor to disk"""
    for label in self.labels:
        if not isinstance(self.predictors[label], str):
            self.predictors[label] = self.predictors[label].path
    save_pkl.save(path=os.path.join(self.path, self.multi_predictor_file), object=self)
    print(f"MultilabelPredictor saved to disk. Load with: MultilabelPredictor.load('{self.path}')")

@classmethod
def load(cls, path):
    """Load MultilabelPredictor from disk"""
    path = os.path.expanduser(path)
    return load_pkl.load(path=os.path.join(path, cls.multi_predictor_file))

def get_predictor(self, label):
    """Returns TabularPredictor for specific label"""
    predictor = self.predictors[label]
    if isinstance(predictor, str):
        return TabularPredictor.load(path=predictor)
    return predictor
```

## Usage Example

### Training
```python
# Define data and parameters
train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
subsample_size = 500  # For faster demo
train_data = train_data.sample(n=subsample_size, random_state=0)

labels = ['education-num','education','class']
problem_types = ['regression','multiclass','binary']
eval_metrics = ['mean_absolute_error','accuracy','accuracy']
save_path = 'agModels-predictEducationClass'
time_limit = 5  # seconds per label

# Create and train the multi-label predictor
multi_predictor = MultilabelPredictor(labels=labels, problem_types=problem_types, 
                                     eval_metrics=eval_metrics, path=save_path)
multi_predictor.fit(train_data, time_limit=time_limit)
```

### Inference and Evaluation
```python
# Load test data
test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')
test_data = test_data.sample(n=subsample_size, random_state=0)
test_data_nolab = test_data.drop(columns=labels)

# Load model and make predictions
multi_predictor = MultilabelPredictor.load(save_path)
predictions = multi_predictor.predict(test_data_nolab)

# Evaluate performance
evaluations = multi_predictor.evaluate(test_data)
```

### Accessing Individual Predictors
```python
predictor_class = multi_predictor.get_predictor('class')
predictor_class.leaderboard()
```

## Best Practices

1. Specify `eval_metrics` for each label to optimize performance
2. Use `presets='best_quality'` for better predictive performance
3. Set `consider_labels_correlation=False` if you plan to use individual predictors separately
4. For memory issues, use techniques from the TabularPredictor in-depth tutorial
5. For faster inference, try `presets = ['good_quality', 'optimize_for_deployment']`