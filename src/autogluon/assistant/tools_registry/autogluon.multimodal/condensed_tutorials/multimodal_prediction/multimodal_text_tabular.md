# Condensed: ```python

Summary: "

Summary: Summary: "

Summary: "

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: 

Summary: This tutorial demonstrates how to implement book price prediction using AutoGluon MultiModal, showcasing techniques for handling mixed data types (text and numeric features), preprocessing text data, and applying log transformation for regression tasks. It covers training a multimodal predictor with minimal configuration, making predictions, evaluating model performance, and extracting embeddings for downstream tasks. Key features include automatic handling of mixed data types, simple API for model training and prediction, and embedding extraction capabilities—all valuable for developing price prediction systems with textual and numerical inputs.

*This is a condensed version that preserves essential implementation details and context.*

# Book Price Prediction with AutoGluon MultiModal

## Setup

```python
!pip install autogluon.multimodal openpyxl

import numpy as np
import pandas as pd
import warnings
import os

warnings.filterwarnings('ignore')
np.random.seed(123)
```

## Data Preparation

```python
# Download and extract dataset
!mkdir -p price_of_books
!wget https://automl-mm-bench.s3.amazonaws.com/machine_hack_competitions/predict_the_price_of_books/Data.zip -O price_of_books/Data.zip
!cd price_of_books && unzip -o Data.zip

# Load data
train_df = pd.read_excel(os.path.join('price_of_books', 'Participants_Data', 'Data_Train.xlsx'), engine='openpyxl')

# Preprocessing function
def preprocess(df):
    df = df.copy(deep=True)
    df.loc[:, 'Reviews'] = pd.to_numeric(df['Reviews'].apply(lambda ele: ele[:-len(' out of 5 stars')]))
    df.loc[:, 'Ratings'] = pd.to_numeric(df['Ratings'].apply(lambda ele: ele.replace(',', '')[:-len(' customer reviews')]))
    df.loc[:, 'Price'] = np.log(df['Price'] + 1)  # Log transform price
    return df

# Create train/test splits
train_subsample_size = 1500  # Smaller sample for faster demo
test_subsample_size = 5
train_df = preprocess(train_df)
train_data = train_df.iloc[100:].sample(train_subsample_size, random_state=123)
test_data = train_df.iloc[:100].sample(test_subsample_size, random_state=245)
```

## Training the Model

```python
from autogluon.multimodal import MultiModalPredictor
import uuid

# Set training parameters
time_limit = 3 * 60  # 3 minutes training time
model_path = f"./tmp/{uuid.uuid4().hex}-automm_text_book_price_prediction"

# Create and train the predictor
predictor = MultiModalPredictor(label='Price', path=model_path)
predictor.fit(train_data, time_limit=time_limit)
```

## Prediction and Evaluation

```python
# Make predictions
predictions = predictor.predict(test_data)
print('Predictions:')
print(np.exp(predictions) - 1)  # Convert back from log scale
print('\nTrue Value:')
print(np.exp(test_data['Price']) - 1)

# Evaluate model performance
performance = predictor.evaluate(test_data)
print(performance)

# Extract embeddings
embeddings = predictor.extract_embedding(test_data)
print(embeddings.shape)
```

## Key Points

- AutoGluon MultiModal automatically handles mixed data types (text, numeric features)
- Log transformation of price values helps with regression performance
- The model can be trained with minimal configuration using `predictor.fit()`
- For real applications, increase `time_limit` and use more training data
- Embeddings can be extracted for downstream tasks using `predictor.extract_embedding()`

For customization options, refer to the "Customize AutoMM" documentation.