# Condensed: ```python

Summary: This tutorial demonstrates using AutoGluon Multimodal for multimodal classification on the PetFinder dataset. It covers: (1) implementing multimodal machine learning with images and tabular data, including proper image path handling and feature metadata configuration; (2) solving pet adoption prediction tasks by integrating different data types; and (3) key functionalities including data preparation for multimodal inputs, configuring feature metadata to identify image columns, creating appropriate hyperparameter configurations, and training a TabularPredictor that automatically handles mixed data types. The tutorial showcases how to process image paths, configure models for multimodal inputs, and evaluate performance on test data.

*This is a condensed version that preserves essential implementation details and context.*

# AutoGluon Multimodal Tutorial: PetFinder Dataset

## Setup and Data Preparation

```python
!pip install autogluon

# Download and extract dataset
download_dir = './ag_petfinder_tutorial'
zip_file = 'https://automl-mm-bench.s3.amazonaws.com/petfinder_kaggle.zip'

from autogluon.core.utils.loaders import load_zip
load_zip.unzip(zip_file, unzip_dir=download_dir)

# Load data
import pandas as pd
dataset_path = download_dir + '/petfinder_processed'
train_data = pd.read_csv(f'{dataset_path}/train.csv', index_col=0)
test_data = pd.read_csv(f'{dataset_path}/dev.csv', index_col=0)

# Define key columns
label = 'AdoptionSpeed'  # Target variable (multi-class classification)
image_col = 'Images'
```

## Image Column Preprocessing

```python
# Extract only the first image per row (AutoGluon supports one image per row)
train_data[image_col] = train_data[image_col].apply(lambda ele: ele.split(';')[0])
test_data[image_col] = test_data[image_col].apply(lambda ele: ele.split(';')[0])

# Update image paths to absolute paths
def path_expander(path, base_folder):
    path_l = path.split(';')
    return ';'.join([os.path.abspath(os.path.join(base_folder, path)) for path in path_l])

train_data[image_col] = train_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
test_data[image_col] = test_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
```

## Sample Data for Faster Prototyping

```python
# For tutorial purposes - use smaller dataset
train_data = train_data.sample(500, random_state=0)
```

## Feature Metadata Configuration

```python
from autogluon.tabular import FeatureMetadata
# Create feature metadata and specify image column
feature_metadata = FeatureMetadata.from_df(train_data)
feature_metadata = feature_metadata.add_special_types({image_col: ['image_path']})
```

## Model Configuration and Training

```python
# Get multimodal hyperparameter configuration
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config
hyperparameters = get_hyperparameter_config('multimodal')
# This config includes tabular models, Electra BERT text model, and ResNet image model

# Train the predictor
from autogluon.tabular import TabularPredictor
predictor = TabularPredictor(label=label).fit(
    train_data=train_data,
    hyperparameters=hyperparameters,
    feature_metadata=feature_metadata,
    time_limit=900,  # 15 minutes time limit
)

# Evaluate on test data
leaderboard = predictor.leaderboard(test_data)
```

## Key Notes

- AutoGluon automatically identifies text columns but requires manual specification of image columns
- When prototyping with large multimodal datasets, start with a small sample to identify promising models
- The 'multimodal' preset handles tabular, text, and image data simultaneously
- Training on multimodal data can be computationally intensive, especially with 'best_quality' preset