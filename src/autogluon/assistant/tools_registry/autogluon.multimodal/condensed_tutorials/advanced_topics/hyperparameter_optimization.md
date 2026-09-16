# Condensed: ```python

Summary: This tutorial demonstrates hyperparameter optimization (HPO) for AutoGluon's MultiModalPredictor using Ray Tune backend for image classification tasks. It covers baseline model fitting with default settings, defining search spaces using `tune.uniform()` and `tune.choice()` for parameters like learning rate, optimizer type, epochs, and model checkpoints (e.g., ghostnet_100, mobilenetv3). Key implementation details include configuring `hyperparameter_tune_kwargs` with searcher options (random/bayes), schedulers (FIFO/ASHA), trial counts, and checkpoint management. The tutorial helps with tasks involving automated model selection, performance optimization through systematic hyperparameter search, and evaluating models with custom metrics. Essential for implementing production-ready multimodal ML pipelines with automated tuning capabilities.

*This is a condensed version that preserves essential implementation details and context.*

# AutoGluon MultiModal HPO Tutorial (Condensed)

## Installation & Setup

```python
!pip install autogluon.multimodal

import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
from autogluon.multimodal.utils.misc import shopee_dataset

download_dir = './ag_automm_tutorial_hpo'
train_data, test_data = shopee_dataset(download_dir)
train_data = train_data.sample(frac=0.5)  # 400 data points total
```

Dataset has `image` column (file paths) and `label` column (classes).

## Regular Model Fitting (Baseline)

```python
from autogluon.multimodal import MultiModalPredictor

predictor_regular = MultiModalPredictor(label="label")
predictor_regular.fit(
    train_data=train_data,
    hyperparameters={"model.timm_image.checkpoint_name": "ghostnet_100"}
)

scores = predictor_regular.evaluate(test_data, metrics=["accuracy"])
print('Top-1 test acc: %.3f' % scores["accuracy"])
```

## HPO Configuration

AutoMM uses [Ray Tune](https://docs.ray.io/en/latest/tune/index.html) backend. Accepts Tune or AutoGluon search spaces.

### 1. Define Hyperparameter Search Space

```python
from ray import tune

hyperparameters = {
    "optim.lr": tune.uniform(0.00005, 0.005),
    "optim.optim_type": tune.choice(["adamw", "sgd"]),
    "optim.max_epochs": tune.choice(["10", "20"]), 
    "model.timm_image.checkpoint_name": tune.choice([
        "swin_base_patch4_window7_224", 
        "convnext_base_in22ft1k"
    ])
}
```

**Note:** See [Customize AutoMM](customization.ipynb) for full supported parameters.

### 2. Configure HPO Strategy

```python
hyperparameter_tune_kwargs = {
    "searcher": "bayes",      # Options: "random", "bayes"
    "scheduler": "ASHA",      # Options: "FIFO", "ASHA"
    "num_trials": 20,         # Number of HPO trials
    "num_to_keep": 3,         # Checkpoints per trial (≥1, default=3)
}
```

**Important:** `num_to_keep` controls disk usage. See [Ray CheckpointConfig docs](https://docs.ray.io/en/latest/train/api/doc/ray.train.CheckpointConfig.html).

## HPO Implementation Example

```python
predictor_hpo = MultiModalPredictor(label="label")

hyperparameters = {
    "optim.lr": tune.uniform(0.00005, 0.001),
    "model.timm_image.checkpoint_name": tune.choice([
        "ghostnet_100",
        "mobilenetv3_large_100"
    ])
}

hyperparameter_tune_kwargs = {
    "searcher": "bayes",
    "scheduler": "ASHA",
    "num_trials": 2,
    "num_to_keep": 3,
}

predictor_hpo.fit(
    train_data=train_data,
    hyperparameters=hyperparameters,
    hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
)

scores_hpo = predictor_hpo.evaluate(test_data, metrics=["accuracy"])
print('Top-1 test acc: %.3f' % scores_hpo["accuracy"])
```

### Expected Output

Training log shows best trial:
```
Current best trial: 47aef96a with val_accuracy=0.862500011920929 
and parameters={'optim.lr': 0.0007195214018085505, 
'model.timm_image.checkpoint_name': 'ghostnet_100'}
```

**Result:** HPO improves test accuracy by selecting optimal hyperparameter combinations based on validation accuracy.

## Additional Resources

- **More Examples:** [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm)
- **Customization:** [Customize AutoMM](customization.ipynb)