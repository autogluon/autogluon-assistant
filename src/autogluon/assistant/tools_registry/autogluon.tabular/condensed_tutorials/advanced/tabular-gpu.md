# Condensed: ```python

Summary: This tutorial covers GPU resource management in AutoGluon's TabularPredictor for machine learning tasks. It demonstrates how to enable GPU acceleration at predictor level (`num_gpus` parameter) or per-model level (via `hyperparameters` with `ag_args_fit`), configure multi-modal models for tabular/text/image data, and install LightGBM with GPU support. Key implementation knowledge includes hierarchical resource allocation across three levels (predictor, bagged model, base model) using `num_cpus`/`num_gpus`, `ag_args_ensemble`, and `ag_args_fit` parameters. The tutorial helps with tasks like optimizing parallel training with HPO trials and bagging, calculating resource distribution for concurrent model training, and configuring CPU/GPU allocation for ensemble methods with proper resource constraints.

*This is a condensed version that preserves essential implementation details and context.*

# GPU Configuration in AutoGluon

## Basic GPU Allocation

Enable GPU for entire predictor:

```python
predictor = TabularPredictor(label=label).fit(
    train_data,
    num_gpus=1,  # Grant 1 gpu for the entire Tabular Predictor
)
```

Enable GPU for specific models via `hyperparameters`:

```python
hyperparameters = {
    'GBM': [
        {'ag_args_fit': {'num_gpus': 0}},  # Train with CPU
        {'ag_args_fit': {'num_gpus': 1}}   # Train with GPU. Must be <= total num_gpus
    ]
}
predictor = TabularPredictor(label=label).fit(
    train_data, 
    num_gpus=1,
    hyperparameters=hyperparameters, 
)
```

## Multi-modal Configuration

For tabular, text, and image data with limited VRAM:

```python
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config
hyperparameters = get_hyperparameter_config('multimodal')
```

## LightGBM GPU Support

**Warning**: Default LightGBM installation lacks GPU support. Enable via:

```bash
pip uninstall lightgbm -y
pip install lightgbm --install-option=--gpu
```

If this fails, follow the [official GPU installation guide](https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html) including the Python interface section.

## Advanced Resource Allocation

**Resource hierarchy** (each level must be ≤ parent level):
1. **Predictor level**: `num_cpus`, `num_gpus` - total resources
2. **Bagged model level**: `ag_args_ensemble: ag_args_fit: { RESOURCES }` - per bagged model
3. **Base model level**: `ag_args_fit: { RESOURCES }` - per individual model

**Example**: 2 HPO trials, 2 folds, parallel training:

```python
predictor.fit(
    num_cpus=32, num_gpus=4,  # Total resources
    hyperparameters={'NN_TORCH': {}},
    num_bag_folds=2,
    ag_args_ensemble={
        'ag_args_fit': {
            'num_cpus': 10, 'num_gpus': 2,  # Per bagged model
        }
    },
    ag_args_fit={
        'num_cpus': 4, 'num_gpus': 0.5,  # Per base model
    },
    hyperparameter_tune_kwargs={
        'searcher': 'random',
        'scheduler': 'local',
        'num_trials': 2
    }
)
```

**Resource calculation**:
- 2 HPO trials × 10 CPUs × 2 GPUs = 20 CPUs, 4 GPUs (bagged level)
- 2 folds × 4 CPUs × 0.5 GPUs = 8 CPUs, 1 GPU per trial (base level)
- **Total usage**: 16 CPUs, 2 GPUs with 4 models training in parallel