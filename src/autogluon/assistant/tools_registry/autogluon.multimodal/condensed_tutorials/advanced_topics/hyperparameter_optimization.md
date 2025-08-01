# Condensed: ```python

Summary: This tutorial demonstrates hyperparameter optimization (HPO) for multimodal models using AutoGluon's MultiModalPredictor. It covers implementing Ray Tune-based HPO for image classification tasks by defining search spaces for learning rates, optimizers, epochs, and model architectures. Key features include configuring search strategies (random/Bayesian), schedulers (FIFO/ASHA), and trial management. The tutorial shows how to compare regular training with HPO approaches, evaluate model performance, and efficiently tune hyperparameters to improve accuracy. This knowledge helps with implementing automated hyperparameter tuning for computer vision tasks, particularly when working with limited training data.

*This is a condensed version that preserves essential implementation details and context.*

# AutoGluon MultiModal Hyperparameter Optimization Tutorial

## Setup and Data Preparation

```python
!pip install autogluon.multimodal

import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
from autogluon.multimodal.utils.misc import shopee_dataset

# Load and prepare dataset
download_dir = './ag_automm_tutorial_hpo'
train_data, test_data = shopee_dataset(download_dir)
train_data = train_data.sample(frac=0.5)  # 400 data points total
```

The dataset contains image paths in the `image` column and class labels in the `label` column.

## Basic Model Training

```python
from autogluon.multimodal import MultiModalPredictor

predictor_regular = MultiModalPredictor(label="label")
start_time = datetime.now()
predictor_regular.fit(
    train_data=train_data,
    hyperparameters={"model.timm_image.checkpoint_name": "ghostnet_100"}
)
end_time = datetime.now()
elapsed_seconds = (end_time - start_time).total_seconds()
print("Total fitting time: ", f"{int(elapsed_seconds//60)}m{int(elapsed_seconds%60)}s")

# Evaluate model
scores = predictor_regular.evaluate(test_data, metrics=["accuracy"])
print('Top-1 test acc: %.3f' % scores["accuracy"])
```

## Hyperparameter Optimization (HPO)

AutoGluon uses Ray Tune for HPO with the following key configuration options:

1. **Search Space Definition**:
   ```python
   hyperparameters = {
       "optim.lr": tune.uniform(0.00005, 0.005),
       "optim.optim_type": tune.choice(["adamw", "sgd"]),
       "optim.max_epochs": tune.choice(["10", "20"]),
       "model.timm_image.checkpoint_name": tune.choice(["swin_base_patch4_window7_224", "convnext_base_in22ft1k"])
   }
   ```

2. **HPO Configuration**:
   - `"searcher"`: Search strategy (`"random"` or `"bayes"`)
   - `"scheduler"`: Job scheduling method (`"FIFO"` or `"ASHA"`)
   - `"num_trials"`: Number of HPO trials to run
   - `"num_to_keep"`: Number of checkpoints to keep per trial (must be ≥ 1)

### Example HPO Implementation

```python
from ray import tune

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

start_time_hpo = datetime.now()
predictor_hpo.fit(
    train_data=train_data,
    hyperparameters=hyperparameters,
    hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
)
end_time_hpo = datetime.now()
elapsed_seconds_hpo = (end_time_hpo - start_time_hpo).total_seconds()
print("Total fitting time: ", f"{int(elapsed_seconds_hpo//60)}m{int(elapsed_seconds_hpo%60)}s")

# Evaluate HPO model
scores_hpo = predictor_hpo.evaluate(test_data, metrics=["accuracy"])
print('Top-1 test acc: %.3f' % scores_hpo["accuracy"])
```

During training, you'll see the best trial information in the logs:
```
Current best trial: 47aef96a with val_accuracy=0.862500011920929 and parameters={'optim.lr': 0.0007195214018085505, 'model.timm_image.checkpoint_name': 'ghostnet_100'}
```

Even with just 2 trials, HPO can improve performance by finding better hyperparameter combinations compared to default settings.

## Additional Resources
- More examples: [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm)
- For customization: See [Customize AutoMM](customization.ipynb)