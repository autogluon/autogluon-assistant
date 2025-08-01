# Condensed: ```python

Summary: This tutorial explains GPU acceleration in AutoGluon, covering basic GPU allocation with `num_gpus` parameter, model-specific GPU assignment using hyperparameters dictionary, and multi-modal configuration retrieval. It details special installation requirements for GPU-enabled LightGBM and demonstrates advanced resource allocation techniques to control CPU/GPU usage at predictor, ensemble, and individual model levels. The tutorial helps with optimizing machine learning workflows by efficiently distributing computational resources across different models and training processes, particularly useful for implementing parallel hyperparameter optimization with controlled resource allocation for tabular, multimodal, and gradient-boosted models.

*This is a condensed version that preserves essential implementation details and context.*

# GPU Acceleration in AutoGluon

## Basic GPU Usage

```python
predictor = TabularPredictor(label=label).fit(
    train_data,
    num_gpus=1,  # Grant 1 GPU for the entire Tabular Predictor
)
```

## Model-Specific GPU Allocation

```python
hyperparameters = {
    'GBM': [
        {'ag_args_fit': {'num_gpus': 0}},  # Train with CPU
        {'ag_args_fit': {'num_gpus': 1}}   # Train with GPU
    ]
}
predictor = TabularPredictor(label=label).fit(
    train_data, 
    num_gpus=1,
    hyperparameters=hyperparameters, 
)
```

## Multi-modal Configuration

For multimodal data (tabular, text, and image), you can retrieve the default configuration:

```python
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config
hyperparameters = get_hyperparameter_config('multimodal')
```

## Enabling GPU for LightGBM

LightGBM requires special installation for GPU support:
1. Uninstall existing LightGBM: `pip uninstall lightgbm -y`
2. Install GPU version: `pip install lightgbm --install-option=--gpu`

If this doesn't work, follow the [official guide](https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html) to install from source.

## Advanced Resource Allocation

Control resources at different levels:
- `num_cpus` and `num_gpus` at predictor level
- `ag_args_ensemble: ag_args_fit: { RESOURCES }` for bagged models
- `ag_args_fit: { RESOURCES }` for individual base models

Example with detailed resource allocation:

```python
predictor.fit(
    num_cpus=32,
    num_gpus=4,
    hyperparameters={'NN_TORCH': {}},
    num_bag_folds=2,
    ag_args_ensemble={
        'ag_args_fit': {
            'num_cpus': 10,
            'num_gpus': 2,
        }
    },
    ag_args_fit={
        'num_cpus': 4,
        'num_gpus': 0.5,
    },
    hyperparameter_tune_kwargs={
        'searcher': 'random',
        'scheduler': 'local',
        'num_trials': 2
    }
)
```

This configuration runs 2 HPO trials in parallel, each with 2 parallel folds, using a total of 16 CPUs and 2 GPUs.