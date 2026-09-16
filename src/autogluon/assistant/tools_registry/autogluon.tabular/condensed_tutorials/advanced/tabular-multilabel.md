# Condensed: ```python

Summary: This tutorial demonstrates implementing a custom `MultilabelPredictor` wrapper class in AutoGluon for simultaneous multi-column prediction tasks. It covers creating multiple `TabularPredictor` instances with configurable problem types (regression, multiclass, binary), implementing auto-regressive or independent label prediction via `consider_labels_correlation`, and handling mixed prediction problems in tabular data. Key techniques include iterative model training with selective column dropping, model persistence/loading, batch prediction across multiple labels, and accessing individual predictors. Useful for building multi-output ML systems, handling correlated target variables, and optimizing multi-label workflows with presets for quality vs. performance trade-offs.

*This is a condensed version that preserves essential implementation details and context.*

# Multi-Label Prediction with AutoGluon

## Installation & Setup

```python
!pip install autogluon.tabular[all]
```

## MultilabelPredictor Class

A wrapper that creates multiple `TabularPredictor` objects to predict multiple columns simultaneously.

**Key Parameters:**
- `labels`: List of column names to predict
- `path`: Directory for saving models (auto-generated if unspecified)
- `problem_types`: List of problem types for each label (optional)
- `eval_metrics`: List of evaluation metrics for each label (optional)
- `consider_labels_correlation`: If `True`, predicts labels auto-regressively (order matters). Set to `False` for independent predictions

**Core Implementation:**

```python
class MultilabelPredictor:
    def __init__(self, labels, path=None, problem_types=None, eval_metrics=None, 
                 consider_labels_correlation=True, **kwargs):
        self.path = setup_outputdir(path, warn_if_exist=False)
        self.labels = labels
        self.consider_labels_correlation = consider_labels_correlation
        self.predictors = {}
        
        for i in range(len(labels)):
            label = labels[i]
            path_i = os.path.join(self.path, "Predictor_" + str(label))
            self.predictors[label] = TabularPredictor(
                label=label, 
                problem_type=problem_types[i] if problem_types else None,
                eval_metric=eval_metrics[i] if eval_metrics else None,
                path=path_i, **kwargs)
```

## Training

```python
# Load and prepare data
train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
train_data = train_data.sample(n=500, random_state=0)

# Configure multi-label prediction
labels = ['education-num', 'education', 'class']
problem_types = ['regression', 'multiclass', 'binary']
eval_metrics = ['mean_absolute_error', 'accuracy', 'accuracy']

# Train
multi_predictor = MultilabelPredictor(labels=labels, problem_types=problem_types, 
                                      eval_metrics=eval_metrics, path='agModels-predictEducationClass')
multi_predictor.fit(train_data, time_limit=5)
```

**Fit Implementation Detail:**
```python
def fit(self, train_data, tuning_data=None, **kwargs):
    for i in range(len(self.labels)):
        label = self.labels[i]
        if not self.consider_labels_correlation:
            labels_to_drop = [l for l in self.labels if l != label]
        else:
            labels_to_drop = [self.labels[j] for j in range(i+1, len(self.labels))]
        train_data = train_data_og.drop(labels_to_drop, axis=1)
        predictor.fit(train_data=train_data, tuning_data=tuning_data, **kwargs)
```

## Inference & Evaluation

```python
# Load saved predictor
multi_predictor = MultilabelPredictor.load('agModels-predictEducationClass')

# Predict
test_data_nolab = test_data.drop(columns=labels)
predictions = multi_predictor.predict(test_data_nolab)

# Evaluate (requires ground truth labels)
evaluations = multi_predictor.evaluate(test_data)
```

**Access individual predictor:**
```python
predictor_class = multi_predictor.get_predictor('class')
predictor_class.leaderboard()
```

## Best Practices

**For Best Quality:**
- Specify `eval_metrics` matching your evaluation criteria
- Use `presets='best_quality'` for stack ensembling

**Memory/Disk Issues:**
- Apply strategies from "If you encounter memory/disk issues" in TabularPredictor docs

**Faster Inference:**
- Use `presets=['good_quality', 'optimize_for_deployment']`
- Set `consider_labels_correlation=False` if using individual predictors

**⚠️ Warning:** When `consider_labels_correlation=True`, label order affects accuracy due to auto-regressive prediction.