Summary: This tutorial covers implementing tabular foundation models in AutoGluon for classification and regression tasks. It provides installation commands for three models (Mitra, TabICL, TabPFNv2) with specific model keys and hyperparameter configurations. Key implementations include: zero-shot and fine-tuning workflows with Mitra (`fine_tune`, `fine_tune_steps` parameters), TabICL for large datasets (up to 500K rows), TabPFNv2 for small datasets (<10K rows), and ensemble methods combining multiple foundation models. The tutorial demonstrates `TabularPredictor` usage with `hyperparameters` dict, `problem_type` specification for regression, model evaluation via `leaderboard()`, and probability predictions with `predict_proba()`. Critical constraints include dataset size limits per model and licensing requirements (commercial vs. open-source options).

## Tabular foundation models at a glance

The table below lists the tabular foundation models AutoGluon can fit, in release order. The row,
feature and class limits are the fit constraints AutoGluon applies (`ag.max_rows`, `ag.max_features`,
`ag.max_classes`): above them the model is skipped rather than fitted, so a portfolio degrades to its
other models instead of failing. A dash means AutoGluon sets no limit.

| Model | Key | Released | Max rows | Max features | Max classes | Tasks | License |
|---|---|---|---:|---:|---:|---|---|
| TabPFN-1 | *not in AutoGluon* | 2022-07 | 1,000 | 100 | 10 | classification | Apache-2.0 |
| TabDPT | `TABDPT` | 2024-10 | 100,000 | 2,500 | 160 | all | Apache-2.0 |
| TabPFNv2 | `REALTABPFN-V2` | 2025-01 | 10,000 | 500 | 10 | all | Prior Labs License (commercial use permitted) |
| TabICL | `TABICL` (v1 checkpoint) | 2025-02 | 500,000 | 2,000 | — | all | BSD-3-Clause |
| Mitra | `MITRA` | 2025-07 | 10,000 | 500 | 10 | all | Apache-2.0 |
| RealTabPFN-2.5 | `REALTABPFN-V2.5` | 2025-11 | 100,000 | 2,000 | 10 | all | Commercial license required |
| TabICLv2 | `TABICL` (default) | 2026-02 | 500,000 | 2,000 | — | all | BSD-3-Clause |
| TabPFN-2.6 | `TABPFN-2.6` | 2026-03 | 100,000 | — | 10 | all | Commercial license required |
| TabPFN-3 | `TABPFN-3` | 2026-05 | 500,000 | — | 160 | all | Commercial license required |
| TabDPT-Turbo | `TABDPT-TURBO` | 2026-06 | 100,000 | — | 160 | all | Apache-2.0 |
| Nori | `NORI` | 2026-06 | 50,000 | — | — | regression only | Apache-2.0 |

Notes:
- **Licensing is the first thing to check.** The TabPFN models from 2.5 onwards are free for research
  and internal experimentation, but any commercial use (production, client work, or benchmarking that
  informs a business decision) requires a license or API agreement from Prior Labs
  ([license FAQ](https://docs.priorlabs.ai/models#tabpfn-model-license)). TabPFNv2, TabICL, TabDPT, Mitra and Nori are all free for commercial use.
- **Nori-30M** is a larger Nori checkpoint (2026-07), selected with `{'model': 'nori-30m'}`.
- **TabICL** defaults to the v2 checkpoints; v1 is reachable through the `checkpoint_version`
  hyperparameter.
- Limits are per fit, measured on the training split, and several models are far slower near their
  upper bound than well below it.
- Two AutoGluon presets bundle these models into a portfolio rather than fitting one at a time:
  `extreme_quality` (models free for commercial use only) and `noncommercial` (adds TabPFN-3).


## Installation

First, let's install AutoGluon with support for foundational models:


```python
# Individual model installations:
!pip install uv
!uv pip install autogluon.tabular[mitra]   # For Mitra
!uv pip install autogluon.tabular[tabicl]   # For TabICL
!uv pip install autogluon.tabular[tabpfn]   # For TabPFNv2

```


```python
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine, fetch_california_housing
```

## Example Data

For this tutorial, we'll demonstrate the foundational models on three different datasets to showcase their versatility:

1. **Wine Dataset** (Multi-class Classification) - Medium-sized dataset for comparing model performance
3. **California Housing** (Regression) - Regression dataset

Let's load and prepare these datasets:


```python
# Load datasets

# 1. Wine (Multi-class Classification)
wine_data = load_wine()
wine_df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
wine_df['target'] = wine_data.target

# 2. California Housing (Regression)
housing_data = fetch_california_housing()
housing_df = pd.DataFrame(housing_data.data, columns=housing_data.feature_names)
housing_df['target'] = housing_data.target

print("Dataset shapes:")
print(f"Wine: {wine_df.shape}")
print(f"California Housing: {housing_df.shape}")
```

## Create Train/Test Splits

Let's create train/test splits for our datasets:


```python
# Create train/test splits (80/20)
wine_train, wine_test = train_test_split(wine_df, test_size=0.2, random_state=42, stratify=wine_df['target'])
housing_train, housing_test = train_test_split(housing_df, test_size=0.2, random_state=42)

print("Training set sizes:")
print(f"Wine: {len(wine_train)} samples")
print(f"Housing: {len(housing_train)} samples")

# Convert to TabularDataset
wine_train_data = TabularDataset(wine_train)
wine_test_data = TabularDataset(wine_test)
housing_train_data = TabularDataset(housing_train)
housing_test_data = TabularDataset(housing_test)
```

## 1. Mitra: AutoGluon's Tabular Foundation Model

[Mitra](https://huggingface.co/autogluon/mitra-classifier) is a tabular foundation model developed by the AutoGluon team, natively supported in AutoGluon with just three lines of code via `predictor.fit())`. Built on the in-context learning paradigm and pretrained exclusively on synthetic data, Mitra introduces a principled pretraining approach by carefully selecting and mixing diverse synthetic priors to promote robust generalization across a wide range of real-world tabular datasets.

**Paper (NeurIPS 2025)**: ["Mitra: Mixed Synthetic Priors for Enhancing Tabular Foundation Models"](https://neurips.cc/virtual/2025/poster/115625)  
**Authors**: Xiyuan Zhang, Danielle Maddix Robinson, Junming Yin, Nick Erickson, Abdul Fatir Ansari, Boran Han, Shuai Zhang, Leman Akoglu, Christos Faloutsos, Michael Mahoney, Tony Hu, Huzefa Rangwala, George Karypis, Yuyang (Bernie) Wang

📊 **Mitra is strongest on small tabular datasets** with fewer than 5,000 samples and 100 features, for both classification and regression tasks. At the time of its release it was state-of-the-art on TabRepo, TabZilla, AMLB and TabArena, beating TabPFNv2, TabICL, CatBoost and RealMLP; newer foundation models have since overtaken it.

🧠 **Mitra supports both zero-shot and fine-tuning modes** and runs seamlessly on both GPU and CPU. Its weights are fully open-sourced under the Apache-2.0 license, making it a privacy-conscious and production-ready solution for enterprises concerned about data sharing and hosting.

🔗 **Learn more on Hugging Face:**
- Classification model: [autogluon/mitra-classifier](https://huggingface.co/autogluon/mitra-classifier)
- Regression model: [autogluon/mitra-regressor](https://huggingface.co/autogluon/mitra-regressor)

### Using Mitra for Classification


```python
# Create predictor with Mitra
print("Training Mitra classifier on classification dataset...")
mitra_predictor = TabularPredictor(label='target')
mitra_predictor.fit(
    wine_train_data,
    hyperparameters={
        'MITRA': {'fine_tune': False}
    },
   )

print("\nMitra training completed!")
```

## Evaluate Mitra Performance


```python
# Make predictions
mitra_predictions = mitra_predictor.predict(wine_test_data)
print("Sample Mitra predictions:")
print(mitra_predictions.head(10))

# Show prediction probabilities for first few samples
mitra_predictions = mitra_predictor.predict_proba(wine_test_data)
print(mitra_predictions.head())

# Show model leaderboard
print("\nMitra Model Leaderboard:")
mitra_predictor.leaderboard(wine_test_data)

```

## Finetuning with Mitra


```python
mitra_predictor_ft = TabularPredictor(label='target')
mitra_predictor_ft.fit(
    wine_train_data,
    hyperparameters={
        'MITRA': {'fine_tune': True, 'fine_tune_steps': 10}
    },
    time_limit=120,  # 2 minutes
   )

print("\nMitra fine-tuning completed!")
```

## Evaluating Fine-tuned Mitra Performance


```python

# Show model leaderboard
print("\nMitra Model Leaderboard:")
mitra_predictor_ft.leaderboard(wine_test_data)

```

## Using Mitra for Regression


```python

# Create predictor with Mitra for regression
print("Training Mitra regressor on California Housing dataset...")
mitra_reg_predictor = TabularPredictor(
    label='target',
    path='./mitra_regressor_model',
    problem_type='regression'
)
mitra_reg_predictor.fit(
    housing_train_data.sample(1000), # sample 1000 rows
    hyperparameters={
        'MITRA': {'fine_tune': False}
    },
)

# Evaluate regression performance
mitra_reg_predictor.leaderboard(housing_test_data)

```

## 2. TabICL: In-Context Learning for Tabular Data

**TabICL** ("**Tab**ular **I**n-**C**ontext **L**earning") is a foundational model designed specifically for in-context learning on large tabular datasets. AutoGluon fits **TabICLv2** by default, which is faster and more accurate than TabICLv1.

**Paper**: ["TabICL: A Tabular Foundation Model for In-Context Learning on Large Data"](https://arxiv.org/abs/2502.05564)  
**Paper (v2)**: ["TabICLv2: A better, faster, scalable, and open tabular foundation model"](https://arxiv.org/abs/2602.11139)  
**Authors**: Jingang Qu, David Holzmüller, Gaël Varoquaux, Marine Le Morvan  
**GitHub**: https://github.com/soda-inria/tabicl

TabICL leverages transformer architecture with in-context learning capabilities, making it particularly effective for scenarios where you have limited training data but access to related examples.


```python
# Train TabICL on dataset
print("Training TabICL on wine dataset...")
tabicl_predictor = TabularPredictor(
    label='target',
    path='./tabicl_model'
)
tabicl_predictor.fit(
    wine_train_data,
    hyperparameters={
        'TABICL': {},
    },
)

# Show prediction probabilities for first few samples
tabicl_predictions = tabicl_predictor.predict_proba(wine_test_data)
print(tabicl_predictions.head())

# Show TabICL leaderboard
print("\nTabICL Model Details:")
tabicl_predictor.leaderboard(wine_test_data)
```

## 3. TabPFNv2: Prior-Fitted Networks

**TabPFNv2** ("**Tab**ular **P**rior-**F**itted **N**etworks **v2**") is designed for accurate predictions on small tabular datasets by using prior-fitted network architectures. Its AutoGluon model key is `REALTABPFN-V2`, named for the default checkpoints trained on real-world data.

**Paper**: ["Accurate predictions on small data with a tabular foundation model"](https://www.nature.com/articles/s41586-024-08328-6)  
**Authors**: Noah Hollmann, Samuel Müller, Lennart Purucker, Arjun Krishnakumar, Max Körfer, Shi Bin Hoo, Robin Tibor Schirrmeister & Frank Hutter  
**GitHub**: https://github.com/PriorLabs/TabPFN

TabPFNv2 excels on small datasets (< 10,000 samples) by leveraging prior knowledge encoded in the network architecture.

**Newer TabPFN versions**: TabPFN-2.6 (`TABPFN-2.6`) and TabPFN-3 (`TABPFN-3`) install from the same `[tabpfn]` extra and are much stronger than TabPFNv2. Neither is free for commercial use: both require a license or API agreement from Prior Labs ([license FAQ](https://docs.priorlabs.ai/models#tabpfn-model-license)). `REALTABPFN-V2` is the option that is free to use commercially. The `noncommercial` preset fits TabPFN-3 as part of a portfolio.


```python
# Train TabPFNv2 on Wine dataset (perfect size for TabPFNv2)
print("Training TabPFNv2 on Wine dataset...")
tabpfnv2_predictor = TabularPredictor(
    label='target',
    path='./tabpfnv2_model'
)
tabpfnv2_predictor.fit(
    wine_train_data,
    hyperparameters={
        'REALTABPFN-V2': {
            # TabPFNv2 works best with default parameters on small datasets
        },
    },
)

# Show prediction probabilities for first few samples
tabpfnv2_predictions = tabpfnv2_predictor.predict_proba(wine_test_data)
print(tabpfnv2_predictions.head())


tabpfnv2_predictor.leaderboard(wine_test_data)
```

## Advanced Usage: Combining Multiple Foundational Models

AutoGluon allows you to combine multiple foundational models in a single predictor for enhanced performance through model stacking and ensembling:


```python
# Configure multiple foundational models together
multi_foundation_config = {
    'MITRA': {
        'fine_tune': True,
        'fine_tune_steps': 10
    },
    'REALTABPFN-V2': {},
    'TABICL': {},
}

print("Training ensemble of foundational models...")
ensemble_predictor = TabularPredictor(
    label='target',
    path='./ensemble_foundation_model'
).fit(
    wine_train_data,
    hyperparameters=multi_foundation_config,
    time_limit=300,  # More time for multiple models
)

# Evaluate ensemble performance
ensemble_predictor.leaderboard(wine_test_data)

```
