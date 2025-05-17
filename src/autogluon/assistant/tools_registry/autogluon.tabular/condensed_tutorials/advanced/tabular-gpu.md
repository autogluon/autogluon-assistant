# Condensed: Training models with GPU support

Summary: This tutorial demonstrates GPU integration in AutoGluon's TabularPredictor, covering implementation techniques for multi-level resource allocation (predictor, bagged model, and base model levels). It helps with tasks involving GPU-accelerated model training, particularly for LightGBM and neural networks. Key features include configuring single/multiple GPU usage, model-specific GPU allocation, proper CUDA toolkit setup, and hierarchical resource management with specific allocation rules. The tutorial provides practical code examples for both basic and advanced GPU configurations, making it valuable for optimizing machine learning workflows with GPU acceleration.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details:

# Training Models with GPU Support in AutoGluon

## Basic GPU Usage
```python
# Basic GPU allocation
predictor = TabularPredictor(label=label).fit(
    train_data,
    num_gpus=1  # Allocate 1 GPU for TabularPredictor
)

# Model-specific GPU allocation
hyperparameters = {
    'GBM': [
        {'ag_args_fit': {'num_gpus': 0}},  # CPU training
        {'ag_args_fit': {'num_gpus': 1}}   # GPU training
    ]
}
predictor = TabularPredictor(label=label).fit(
    train_data, 
    num_gpus=1,
    hyperparameters=hyperparameters
)
```

## Important Notes
- CUDA toolkit required for GPU training
- LightGBM requires special GPU installation:
  ```bash
  pip uninstall lightgbm -y
  pip install lightgbm --install-option=--gpu
  ```
  If above fails, follow [official guide](https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html)

## Advanced Resource Allocation
Three levels of resource control:
1. TabularPredictor level: `num_cpus`, `num_gpus`
2. Bagged model level: `ag_args_ensemble: ag_args_fit`
3. Base model level: `ag_args_fit`

### Example Configuration
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

### Resource Allocation Rules
- Bagged model resources must be ≤ total TabularPredictor resources
- Base model resources must be ≤ bagged model resources (if applicable)
- Base model resources must be ≤ total TabularPredictor resources