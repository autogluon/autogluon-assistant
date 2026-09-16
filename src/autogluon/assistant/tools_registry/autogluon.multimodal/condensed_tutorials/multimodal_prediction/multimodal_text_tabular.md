# Condensed: ```python

Summary: This tutorial demonstrates AutoGluon MultiModalPredictor for regression tasks on mixed-feature datasets. It covers: (1) data preprocessing techniques including string-to-numeric conversion and log-scale transformations for target variables, (2) automatic neural network generation from heterogeneous features (text, numeric, categorical), (3) simple API usage with MultiModalPredictor.fit() for training with time limits, (4) prediction with automatic inverse transformations, model evaluation, and embedding extraction. Useful for implementing AutoML pipelines on tabular data with mixed data types, price prediction tasks, and scenarios requiring minimal configuration for multimodal learning. Key features include automatic feature type inference, time-constrained training, and built-in evaluation methods.

*This is a condensed version that preserves essential implementation details and context.*

# AutoGluon MultiModal: Book Price Prediction

## Installation & Setup

```python
!pip install autogluon.multimodal
!python3 -m pip install openpyxl

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
np.random.seed(123)
```

## Dataset: Book Price Prediction

Uses [MachineHack Book Price Prediction dataset](https://machinehack.com/hackathons/predict_the_price_of_books/overview) to predict book prices from features like author, abstract, and ratings.

```python
# Download data
!mkdir -p price_of_books
!wget https://automl-mm-bench.s3.amazonaws.com/machine_hack_competitions/predict_the_price_of_books/Data.zip -O price_of_books/Data.zip
!cd price_of_books && unzip -o Data.zip

train_df = pd.read_excel(os.path.join('price_of_books', 'Participants_Data', 'Data_Train.xlsx'), engine='openpyxl')
```

## Preprocessing

**Critical transformations:**
- Convert `Reviews` and `Ratings` to numeric values
- Apply log-scale transformation to prices

```python
def preprocess(df):
    df = df.copy(deep=True)
    df.loc[:, 'Reviews'] = pd.to_numeric(df['Reviews'].apply(lambda ele: ele[:-len(' out of 5 stars')]))
    df.loc[:, 'Ratings'] = pd.to_numeric(df['Ratings'].apply(lambda ele: ele.replace(',', '')[:-len(' customer reviews')]))
    df.loc[:, 'Price'] = np.log(df['Price'] + 1)
    return df

train_subsample_size = 1500  # adjust for production use
test_subsample_size = 5
train_df = preprocess(train_df)
train_data = train_df.iloc[100:].sample(train_subsample_size, random_state=123)
test_data = train_df.iloc[:100].sample(test_subsample_size, random_state=245)
```

## Training

**Key implementation:** Neural network automatically generated based on inferred feature data types.

```python
from autogluon.multimodal import MultiModalPredictor
import uuid

time_limit = 3 * 60  # increase for production
model_path = f"./tmp/{uuid.uuid4().hex}-automm_text_book_price_prediction"
predictor = MultiModalPredictor(label='Price', path=model_path)
predictor.fit(train_data, time_limit=time_limit)
```

## Inference & Evaluation

```python
# Predictions (reverse log transformation)
predictions = predictor.predict(test_data)
print(np.exp(predictions) - 1)  # convert back to original scale
print(np.exp(test_data['Price']) - 1)  # true values

# Evaluation
performance = predictor.evaluate(test_data)

# Extract embeddings
embeddings = predictor.extract_embedding(test_data)
```

## Additional Resources

- [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm)
- [Customize AutoMM](../advanced_topics/customization.ipynb)