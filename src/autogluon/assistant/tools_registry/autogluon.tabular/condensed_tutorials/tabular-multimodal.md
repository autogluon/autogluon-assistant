# Condensed: ```python

Summary: This tutorial demonstrates AutoGluon multimodal classification combining tabular, text, and image data. Key implementations include: preprocessing image columns to single images per row with absolute file paths using `path_expander()`, configuring `FeatureMetadata` with `add_special_types({'image_col': ['image_path']})` to enable image features, and using `get_hyperparameter_config('multimodal')` for BERT+ResNet models. Covers `TabularPredictor.fit()` with custom hyperparameters and feature metadata, data sampling for prototyping, and model evaluation via leaderboard. Essential for tasks requiring multi-modal ML pipelines, image path preprocessing, AutoGluon feature type specification, and combining heterogeneous data sources in classification problems.

*This is a condensed version that preserves essential implementation details and context.*

# AutoGluon Multimodal Tutorial (Condensed)

## Installation & Data Setup

```python
!pip install autogluon

from autogluon.core.utils.loaders import load_zip
download_dir = './ag_petfinder_tutorial'
zip_file = 'https://automl-mm-bench.s3.amazonaws.com/petfinder_kaggle.zip'
load_zip.unzip(zip_file, unzip_dir=download_dir)
```

Dataset structure: CSV files (train/dev/test) + image directories (train_images/test_images)

## Data Loading

```python
import pandas as pd
dataset_path = download_dir + '/petfinder_processed'
train_data = pd.read_csv(f'{dataset_path}/train.csv', index_col=0)
test_data = pd.read_csv(f'{dataset_path}/dev.csv', index_col=0)

label = 'AdoptionSpeed'  # Multi-class classification (5 categories)
image_col = 'Images'
```

## Critical Image Column Preprocessing

**⚠️ Important**: AutoGluon supports **one image per row only**.

```python
# Extract first image from multi-image rows
train_data[image_col] = train_data[image_col].apply(lambda ele: ele.split(';')[0])
test_data[image_col] = test_data[image_col].apply(lambda ele: ele.split(';')[0])

# Expand to absolute paths
def path_expander(path, base_folder):
    path_l = path.split(';')
    return ';'.join([os.path.abspath(os.path.join(base_folder, path)) for path in path_l])

train_data[image_col] = train_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
test_data[image_col] = test_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
```

**Best Practice**: Sample data for prototyping due to computational intensity:
```python
train_data = train_data.sample(500, random_state=0)
```

## FeatureMetadata Configuration

```python
from autogluon.tabular import FeatureMetadata

# AutoGluon auto-detects text columns
feature_metadata = FeatureMetadata.from_df(train_data)

# Manually specify image column with 'image_path' special type
feature_metadata = feature_metadata.add_special_types({image_col: ['image_path']})
```

## Hyperparameters Setup

```python
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config

# Predefined multimodal config: Tabular models + Electra BERT + ResNet
hyperparameters = get_hyperparameter_config('multimodal')
```

## Model Training

```python
from autogluon.tabular import TabularPredictor

predictor = TabularPredictor(label=label).fit(
    train_data=train_data,
    hyperparameters=hyperparameters,
    feature_metadata=feature_metadata,
    time_limit=900,
)

# Evaluate performance
leaderboard = predictor.leaderboard(test_data)
```

## Key Takeaways

- **One image per row** limitation requires preprocessing
- **Absolute file paths** required for image loading
- **FeatureMetadata** with `'image_path'` special type enables image features
- **'multimodal' hyperparameter config** trains tabular, text (BERT), and image (ResNet) models
- **Sampling recommended** for prototyping with `best_quality` preset