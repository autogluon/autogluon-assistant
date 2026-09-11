# Condensed: ```python

Summary: This tutorial demonstrates implementing Chinese Named Entity Recognition (NER) using AutoGluon's MultiModalPredictor. It covers loading Chinese NER datasets, training models with Chinese-specific pretrained checkpoints (hfl/chinese-lert-small), and performing entity extraction. Key implementations include: setting problem_type="ner", configuring hyperparameters for Chinese language models, training with time limits, evaluating model performance, and making predictions on both batch and single-sentence inputs. The tutorial shows visualization techniques for NER results and emphasizes that Chinese NER follows the same workflow as English NER, requiring only appropriate checkpoint selection. Useful for building multilingual NER systems, e-commerce product entity extraction, and Chinese text analysis tasks.

*This is a condensed version that preserves essential implementation details and context.*

# Chinese Named Entity Recognition with AutoGluon

## Installation & Setup

```python
!pip install autogluon.multimodal
```

## Data Loading

```python
from autogluon.multimodal import MultiModalPredictor
from autogluon.core.utils.loaders import load_pd
from autogluon.multimodal.utils import visualize_ner

train_data = load_pd.load('https://automl-mm-bench.s3.amazonaws.com/ner/taobao-ner/chinese_ner_train.csv')
dev_data = load_pd.load('https://automl-mm-bench.s3.amazonaws.com/ner/taobao-ner/chinese_ner_dev.csv')
```

**Entity Labels**: HPPX (brand), HCCX (product), XH (pattern), MISC (miscellaneous info like specifications)

**Visualization**:
```python
visualize_ner(train_data["text_snippet"].iloc[0], train_data["entity_annotations"].iloc[0])
```

## Training

**Key Points**:
- Chinese NER uses same process as English NER
- Select Chinese or multilingual pretrained checkpoint
- Use `hfl/chinese-lert-small` for Chinese text

```python
import uuid

label_col = "entity_annotations"
model_path = f"./tmp/{uuid.uuid4().hex}-automm_ner"

predictor = MultiModalPredictor(problem_type="ner", label=label_col, path=model_path)
predictor.fit(
    train_data=train_data,
    hyperparameters={'model.ner_text.checkpoint_name': 'hfl/chinese-lert-small'},
    time_limit=300  # seconds
)
```

## Evaluation

```python
predictor.evaluate(dev_data)
```

## Prediction

```python
# Batch prediction
output = predictor.predict(dev_data)
visualize_ner(dev_data["text_snippet"].iloc[0], output[0])

# Single sentence prediction
sentence = "2023年兔年挂件新年装饰品小挂饰乔迁之喜门挂小兔子"
predictions = predictor.predict({'text_snippet': [sentence]})
visualize_ner(sentence, predictions[0])
```

## Additional Resources
- [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm)
- [Customize AutoMM](../advanced_topics/customization.ipynb)