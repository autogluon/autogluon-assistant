# Condensed: AutoMM for Text + Tabular - Quick Start

Summary: This tutorial demonstrates implementing multimodal machine learning using AutoGluon's MultiModalPredictor for combined text and tabular data analysis. It covers essential techniques for data preprocessing (including numeric conversions and log transformations), model configuration, training, prediction, and embedding extraction. The tutorial helps with tasks like automated feature fusion, mixed-type data handling, and neural network generation. Key functionalities include automatic architecture selection based on data types, joint training across modalities, embedding extraction for downstream tasks, and flexible model customization options. The implementation focuses on practical aspects like proper numerical preprocessing, time limit setting, and model artifact management.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details:

# AutoMM for Text + Tabular - Quick Start

## Key Implementation Details

### Setup
```python
!pip install autogluon.multimodal openpyxl
import numpy as np
import pandas as pd
from autogluon.multimodal import MultiModalPredictor
```

### Data Preprocessing
```python
def preprocess(df):
    df = df.copy(deep=True)
    # Convert Reviews and Ratings to numeric
    df.loc[:, 'Reviews'] = pd.to_numeric(df['Reviews'].apply(lambda ele: ele[:-len(' out of 5 stars')]))
    df.loc[:, 'Ratings'] = pd.to_numeric(df['Ratings'].apply(lambda ele: ele.replace(',', '')[:-len(' customer reviews')]))
    # Log transform prices
    df.loc[:, 'Price'] = np.log(df['Price'] + 1)
    return df
```

### Training Configuration
```python
# Create predictor
predictor = MultiModalPredictor(
    label='Price',
    path=model_path
)

# Train model
predictor.fit(
    train_data,
    time_limit=180  # 3 minutes
)
```

### Prediction and Embedding Extraction
```python
# Get predictions
predictions = predictor.predict(test_data)

# Evaluate model
performance = predictor.evaluate(test_data)

# Extract embeddings
embeddings = predictor.extract_embedding(test_data)
```

## Key Concepts
- MultiModalPredictor automatically handles mixed data types (text, categorical, numerical)
- Neural network architecture is auto-generated based on feature column types
- Supports joint training across multiple modalities
- Can extract embeddings for downstream tasks

## Best Practices
1. Preprocess numerical features appropriately (e.g., log transform for prices)
2. Set adequate time_limit based on dataset size and complexity
3. Use path parameter to save model artifacts
4. Consider data subsampling for initial experiments

## Important Notes
- Default configuration works well for most cases
- For customization options, refer to "Customize AutoMM" documentation
- Model automatically handles feature fusion across different modalities
- Supports both training and inference on mixed data types