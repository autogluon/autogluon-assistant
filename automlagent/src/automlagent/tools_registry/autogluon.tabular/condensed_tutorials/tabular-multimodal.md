# Condensed: Multimodal Data Tables: Tabular, Text, and Image

Summary: This tutorial demonstrates how to implement multimodal machine learning using AutoGluon, specifically combining tabular, text, and image data for prediction tasks. It provides implementation details for dataset preparation (including handling multiple images), feature metadata configuration, and model training with multimodal hyperparameter presets. The tutorial helps with tasks involving image path preprocessing, feature type specification, and unified model training across different data modalities. Key functionalities covered include handling multiple images per row (selecting first image), path expansion for image files, feature metadata customization, and training configuration using AutoGluon's multimodal preset that incorporates tabular models, BERT for text, and ResNet for images.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on key implementation details:

# Multimodal Data Tables: Tabular, Text, and Image Tutorial

## Key Requirements
- GPU required for image and text models
- Appropriate CUDA versions for Torch installations

## Implementation Steps

### 1. Dataset Preparation
```python
# Download and unzip dataset
download_dir = './ag_petfinder_tutorial'
zip_file = 'https://automl-mm-bench.s3.amazonaws.com/petfinder_kaggle.zip'

from autogluon.core.utils.loaders import load_zip
load_zip.unzip(zip_file, unzip_dir=download_dir)

# Load data
import pandas as pd
train_data = pd.read_csv(f'{dataset_path}/train.csv', index_col=0)
test_data = pd.read_csv(f'{dataset_path}/dev.csv', index_col=0)

label = 'AdoptionSpeed'
image_col = 'Images'
```

### 2. Image Column Preprocessing
```python
# Handle multiple images (keep only first image)
train_data[image_col] = train_data[image_col].apply(lambda ele: ele.split(';')[0])
test_data[image_col] = test_data[image_col].apply(lambda ele: ele.split(';')[0])

# Update image paths
def path_expander(path, base_folder):
    path_l = path.split(';')
    return ';'.join([os.path.abspath(os.path.join(base_folder, path)) for path in path_l])

train_data[image_col] = train_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
test_data[image_col] = test_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
```

### 3. Feature Metadata Configuration
```python
from autogluon.tabular import FeatureMetadata
# Create and customize feature metadata
feature_metadata = FeatureMetadata.from_df(train_data)
feature_metadata = feature_metadata.add_special_types({image_col: ['image_path']})
```

### 4. Hyperparameter Configuration
```python
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config
hyperparameters = get_hyperparameter_config('multimodal')
```

### 5. Model Training
```python
from autogluon.tabular import TabularPredictor
predictor = TabularPredictor(label=label).fit(
    train_data=train_data,
    hyperparameters=hyperparameters,
    feature_metadata=feature_metadata,
    time_limit=900,
)
```

## Important Best Practices
1. When prototyping, sample data to identify effective models before scaling up
2. For large multimodal datasets, start with smaller samples and gradually increase data size
3. Adjust time limits based on dataset size and computational resources
4. AutoGluon currently supports only one image per row

## Key Configurations
- Uses 'multimodal' preset configuration
- Includes:
  - Tabular models
  - Electra BERT text model
  - ResNet image model
- Default time limit: 900 seconds

## Model Evaluation
```python
leaderboard = predictor.leaderboard(test_data)
```

This implementation handles multimodal data combining tabular, text, and image features in a single prediction task.