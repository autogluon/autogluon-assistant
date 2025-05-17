# Condensed: Hyperparameter Optimization in AutoMM

Summary: This tutorial demonstrates hyperparameter optimization (HPO) implementation in AutoGluon's MultiModalPredictor, specifically focusing on configuring and executing HPO for machine learning models. It covers techniques for defining search spaces using Ray Tune, setting up HPO configurations with different searchers (random, Bayesian) and schedulers (FIFO, ASHA), and implementing model fitting with HPO. The tutorial helps with tasks like optimizing learning rates, model checkpoint selection, and training parameters. Key features include integration with Ray Tune backend, support for both AutoGluon and Ray Tune search spaces, checkpoint management, and various searcher/scheduler combinations for efficient hyperparameter tuning.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on key implementation details and practices:

# Hyperparameter Optimization in AutoMM

## Key Implementation Details

### Basic Setup
```python
from autogluon.multimodal import MultiModalPredictor
from ray import tune
```

### Standard Model Fitting
```python
predictor_regular = MultiModalPredictor(label="label")
predictor_regular.fit(
    train_data=train_data,
    hyperparameters={"model.timm_image.checkpoint_name": "ghostnet_100"}
)
```

### HPO Configuration

Two main components for HPO:

1. **Hyperparameter Search Space**
```python
hyperparameters = {
    "optimization.learning_rate": tune.uniform(0.00005, 0.001),
    "model.timm_image.checkpoint_name": tune.choice(["ghostnet_100", "mobilenetv3_large_100"]),
    # Optional additional parameters:
    # "optimization.optim_type": tune.choice(["adamw", "sgd"]),
    # "optimization.max_epochs": tune.choice(["10", "20"])
}
```

2. **HPO Settings**
```python
hyperparameter_tune_kwargs = {
    "searcher": "bayes",  # Options: "random", "bayes"
    "scheduler": "ASHA",  # Options: "FIFO", "ASHA"
    "num_trials": 2,      # Number of HPO trials
    "num_to_keep": 3      # Checkpoints to keep per trial
}
```

### HPO Model Fitting
```python
predictor_hpo = MultiModalPredictor(label="label")
predictor_hpo.fit(
    train_data=train_data,
    hyperparameters=hyperparameters,
    hyperparameter_tune_kwargs=hyperparameter_tune_kwargs
)
```

## Critical Configurations

1. **Search Space Parameters**:
   - Learning rate range
   - Model checkpoints
   - Optimizer types
   - Training epochs

2. **HPO Settings**:
   - Searcher strategy
   - Scheduler type
   - Number of trials
   - Checkpoint management

## Best Practices

1. Define appropriate search spaces based on domain knowledge
2. Balance number of trials with available computing resources
3. Use Bayesian optimization for better efficiency compared to random search
4. Keep sufficient checkpoints (minimum 1) for model recovery

## Important Notes

- HPO uses Ray Tune in the backend
- Supports both Ray Tune and AutoGluon search spaces
- Higher number of trials generally leads to better results but increases computation time
- Monitor "Current best trial" in training logs for optimization progress