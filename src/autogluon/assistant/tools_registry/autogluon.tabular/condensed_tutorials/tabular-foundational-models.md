# Condensed: Installation

Summary: This tutorial demonstrates implementing foundation models for tabular data using AutoGluon. It covers three key models: Mitra (for small datasets with zero-shot and fine-tuning capabilities), TabICL (leveraging in-context learning for limited data), and TabPFNv2 (utilizing prior knowledge for small datasets). The code shows how to prepare data, train individual models for classification and regression tasks, and create ensembles combining multiple foundation models. Developers can learn to implement these specialized tabular models with different configurations, evaluate their performance, and handle both classification and regression problems with small to medium-sized datasets.

*This is a condensed version that preserves essential implementation details and context.*

# Foundation Models for Tabular Data in AutoGluon

## Installation

```python
# Install required packages
!pip install uv
!uv pip install autogluon.tabular[mitra]   # For Mitra
!uv pip install autogluon.tabular[tabicl]   # For TabICL
!uv pip install autogluon.tabular[tabpfn]   # For TabPFNv2

import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine, fetch_california_housing
```

## Data Preparation

```python
# Load datasets
wine_data = load_wine()
wine_df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
wine_df['target'] = wine_data.target

housing_data = fetch_california_housing()
housing_df = pd.DataFrame(housing_data.data, columns=housing_data.feature_names)
housing_df['target'] = housing_data.target

# Create train/test splits (80/20)
wine_train, wine_test = train_test_split(wine_df, test_size=0.2, random_state=42, stratify=wine_df['target'])
housing_train, housing_test = train_test_split(housing_df, test_size=0.2, random_state=42)

# Convert to TabularDataset
wine_train_data = TabularDataset(wine_train)
wine_test_data = TabularDataset(wine_test)
housing_train_data = TabularDataset(housing_train)
housing_test_data = TabularDataset(housing_test)
```

## 1. Mitra: AutoGluon's Tabular Foundation Model

Mitra is a state-of-the-art tabular foundation model that excels on small datasets (<5,000 samples, <100 features) for both classification and regression tasks.

### Zero-Shot Classification

```python
mitra_predictor = TabularPredictor(label='target')
mitra_predictor.fit(
    wine_train_data,
    hyperparameters={
        'MITRA': {'fine_tune': False}
    }
)

# Evaluate
mitra_predictor.leaderboard(wine_test_data)
```

### Fine-tuned Classification

```python
mitra_predictor_ft = TabularPredictor(label='target')
mitra_predictor_ft.fit(
    wine_train_data,
    hyperparameters={
        'MITRA': {'fine_tune': True, 'fine_tune_steps': 10}
    },
    time_limit=120  # 2 minutes
)
```

### Regression

```python
mitra_reg_predictor = TabularPredictor(
    label='target',
    path='./mitra_regressor_model',
    problem_type='regression'
)
mitra_reg_predictor.fit(
    housing_train_data.sample(1000),  # sample 1000 rows
    hyperparameters={
        'MITRA': {'fine_tune': False}
    }
)
```

## 2. TabICL: In-Context Learning for Tabular Data

TabICL leverages transformer architecture with in-context learning capabilities, effective for scenarios with limited training data.

```python
tabicl_predictor = TabularPredictor(
    label='target',
    path='./tabicl_model'
)
tabicl_predictor.fit(
    wine_train_data,
    hyperparameters={
        'TABICL': {},
    }
)
```

## 3. TabPFNv2: Prior-Fitted Networks

TabPFNv2 excels on small datasets (<10,000 samples) by leveraging prior knowledge encoded in the network architecture.

```python
tabpfnv2_predictor = TabularPredictor(
    label='target',
    path='./tabpfnv2_model'
)
tabpfnv2_predictor.fit(
    wine_train_data,
    hyperparameters={
        'TABPFNV2': {}
    }
)
```

## Advanced Usage: Combining Multiple Foundational Models

```python
multi_foundation_config = {
    'MITRA': {
        'fine_tune': True,
        'fine_tune_steps': 10
    },
    'TABPFNV2': {},
    'TABICL': {},
}

ensemble_predictor = TabularPredictor(
    label='target',
    path='./ensemble_foundation_model'
).fit(
    wine_train_data,
    hyperparameters=multi_foundation_config,
    time_limit=300  # More time for multiple models
)

# Evaluate ensemble performance
ensemble_predictor.leaderboard(wine_test_data)
```