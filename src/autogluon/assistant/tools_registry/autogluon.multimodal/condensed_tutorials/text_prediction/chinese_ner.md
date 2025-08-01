# Condensed: ```python

Summary: This tutorial demonstrates Chinese Named Entity Recognition (NER) using AutoGluon MultiModal, covering implementation of NER models specifically for Chinese text. It shows how to load Chinese NER datasets, train a model using a Chinese pretrained checkpoint (hfl/chinese-lert-small), evaluate performance, and visualize predictions. Key functionalities include customizing the model for Chinese language processing, handling entity types like brand, product, pattern, and specifications, and applying NER to e-commerce product descriptions. The workflow demonstrates the complete pipeline from data loading to visualization with minimal code.

*This is a condensed version that preserves essential implementation details and context.*

# Chinese Named Entity Recognition with AutoGluon MultiModal

## Installation
```python
pip install autogluon.multimodal
```

## Loading Data
```python
import autogluton.multimodal
from autogluon.core.utils.loaders import load_pd
from autogluon.multimodal.utils import visualize_ner

# Load dataset
train_data = load_pd.load('https://automl-mm-bench.s3.amazonaws.com/ner/taobao-ner/chinese_ner_train.csv')
dev_data = load_pd.load('https://automl-mm-bench.s3.amazonaws.com/ner/taobao-ner/chinese_ner_dev.csv')
```

## Training
```python
from autogluon.multimodal import MultiModalPredictor
import uuid

label_col = "entity_annotations"
model_path = f"./tmp/{uuid.uuid4().hex}-automm_ner"
predictor = MultiModalPredictor(problem_type="ner", label=label_col, path=model_path)

# Train with Chinese pretrained model
predictor.fit(
    train_data=train_data,
    hyperparameters={'model.ner_text.checkpoint_name':'hfl/chinese-lert-small'},
    time_limit=300, # seconds
)
```

## Evaluation
```python
predictor.evaluate(dev_data)
```

## Prediction and Visualization
```python
# Predict on dev data
output = predictor.predict(dev_data)
visualize_ner(dev_data["text_snippet"].iloc[0], output[0])

# Predict on custom text
sentence = "2023年兔年挂件新年装饰品小挂饰乔迁之喜门挂小兔子"
predictions = predictor.predict({'text_snippet': [sentence]})
visualize_ner(sentence, predictions[0])
```

**Key Notes:**
- Entity types include HPPX (brand), HCCX (product), XH (pattern), and MISC (specifications)
- Chinese NER follows the same process as English NER but requires a Chinese/multilingual foundation model
- The example uses 'hfl/chinese-lert-small' as the backbone model
- For customization options, refer to the AutoMM customization documentation