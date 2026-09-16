# Condensed: Tabular foundation models at a glance

Summary: This tutorial covers implementing tabular foundation models in AutoGluon for classification and regression tasks. It provides installation commands for three models (Mitra, TabICL, TabPFNv2) with specific model keys and hyperparameter configurations. Key implementations include: zero-shot and fine-tuning workflows with Mitra (`fine_tune`, `fine_tune_steps` parameters), TabICL for large datasets (up to 500K rows), TabPFNv2 for small datasets (<10K rows), and ensemble methods combining multiple foundation models. The tutorial demonstrates `TabularPredictor` usage with `hyperparameters` dict, `problem_type` specification for regression, model evaluation via `leaderboard()`, and probability predictions with `predict_proba()`. Critical constraints include dataset size limits per model and licensing requirements (commercial vs. open-source options).

*This is a condensed version that preserves essential implementation details and context.*

# Tabular Foundation Models in AutoGluon

## Model Overview

AutoGluon supports multiple tabular foundation models with varying constraints and licenses:

| Model | Key | Max rows | Max features | Max classes | License |
|---|---|---:|---:|---:|---|
| TabPFNv2 | `REALTABPFN-V2` | 10,000 | 500 | 10 | Prior Labs (commercial use permitted) |
| TabICLv2 | `TABICL` | 500,000 | 2,000 | — | BSD-3-Clause |
| Mitra | `MITRA` | 10,000 | 500 | 10 | Apache-2.0 |
| TabPFN-3 | `TABPFN-3` | 500,000 | — | 160 | Commercial license required |

**⚠️ Licensing**: TabPFN-2.5+ requires commercial license from Prior Labs. TabPFNv2, TabICL, and Mitra are free for commercial use.

**Presets**: `extreme_quality` (commercial-free models only), `noncommercial` (includes TabPFN-3)

## Installation

```python
!pip install uv
!uv pip install autogluon.tabular[mitra]   # For Mitra
!uv pip install autogluon.tabular[tabicl]  # For TabICL
!uv pip install autogluon.tabular[tabpfn]  # For TabPFNv2
```

## 1. Mitra - AutoGluon's Foundation Model

**Best for**: Small datasets (<5,000 samples, <100 features)  
**Supports**: Zero-shot and fine-tuning, GPU/CPU, classification and regression  
**Paper**: [NeurIPS 2025](https://neurips.cc/virtual/2025/poster/115625)

### Classification Example

```python
from autogluon.tabular import TabularPredictor

# Zero-shot inference
mitra_predictor = TabularPredictor(label='target')
mitra_predictor.fit(
    train_data,
    hyperparameters={'MITRA': {'fine_tune': False}}
)

# Fine-tuning
mitra_predictor_ft = TabularPredictor(label='target')
mitra_predictor_ft.fit(
    train_data,
    hyperparameters={
        'MITRA': {'fine_tune': True, 'fine_tune_steps': 10}
    },
    time_limit=120
)
```

### Regression Example

```python
mitra_reg_predictor = TabularPredictor(
    label='target',
    problem_type='regression'
)
mitra_reg_predictor.fit(
    train_data,
    hyperparameters={'MITRA': {'fine_tune': False}}
)
```

## 2. TabICL - In-Context Learning

**Best for**: Large tabular datasets with in-context learning  
**Default**: TabICLv2 (faster, more accurate than v1)  
**Papers**: [v1](https://arxiv.org/abs/2502.05564), [v2](https://arxiv.org/abs/2602.11139)

```python
tabicl_predictor = TabularPredictor(label='target')
tabicl_predictor.fit(
    train_data,
    hyperparameters={'TABICL': {}}
)

# Access v1 checkpoints
tabicl_v1_predictor.fit(
    train_data,
    hyperparameters={'TABICL': {'checkpoint_version': 'v1'}}
)
```

## 3. TabPFNv2 - Prior-Fitted Networks

**Best for**: Small datasets (<10,000 samples)  
**Key**: `REALTABPFN-V2` (commercial-friendly)  
**Paper**: [Nature 2024](https://www.nature.com/articles/s41586-024-08328-6)

```python
tabpfnv2_predictor = TabularPredictor(label='target')
tabpfnv2_predictor.fit(
    train_data,
    hyperparameters={'REALTABPFN-V2': {}}
)
```

**⚠️ Newer versions**: `TABPFN-2.6` and `TABPFN-3` are stronger but require commercial licenses.

## Ensemble Multiple Foundation Models

```python
multi_foundation_config = {
    'MITRA': {'fine_tune': True, 'fine_tune_steps': 10},
    'REALTABPFN-V2': {},
    'TABICL': {},
}

ensemble_predictor = TabularPredictor(label='target').fit(
    train_data,
    hyperparameters=multi_foundation_config,
    time_limit=300
)

ensemble_predictor.leaderboard(test_data)
```

## Key Notes

- **Limits are per fit** on training split; models skip if exceeded
- **Performance degrades** near upper bounds
- Use `predictor.leaderboard(test_data)` to evaluate
- Use `predict_proba()` for classification probabilities