# Condensed: ```python

Summary: This tutorial demonstrates knowledge distillation using AutoGluon MultiModal to compress large transformer models into smaller, efficient versions. It covers loading GLUE QNLI datasets, preparing train/validation/test splits, loading a pre-trained BERT-base (12-layer) teacher model, and distilling knowledge into a BERT (6-layer) student model by passing `teacher_predictor` to the `.fit()` method. Key implementation details include using `MultiModalPredictor` with `teacher_predictor` parameter, configuring hyperparameters like `model.hf_text.checkpoint_name` for model selection, and evaluating distilled models. This helps with tasks involving model compression, creating efficient NLP models for deployment, transfer learning with teacher-student architectures, and text classification with reduced computational requirements while maintaining performance through prediction/feature map matching.

*This is a condensed version that preserves essential implementation details and context.*

# Knowledge Distillation with AutoGluon MultiModal

## Setup and Data Preparation

```python
!pip install autogluon.multimodal

from datasets import load_dataset
from sklearn.model_selection import train_test_split

# Load GLUE QNLI dataset
dataset = load_dataset("glue", "qnli")

# Prepare train/valid/test splits
train_valid_df = dataset["train"].to_pandas()[["question", "sentence", "label"]].sample(1000, random_state=123)
train_df, valid_df = train_test_split(train_valid_df, test_size=0.2, random_state=123)
test_df = dataset["validation"].to_pandas()[["question", "sentence", "label"]].sample(1000, random_state=123)
```

## Load Teacher Model

Load a pre-trained teacher model with **BERT-base (12-layer)** backbone:

```python
!wget https://automl-mm-bench.s3.amazonaws.com/unit-tests/distillation_sample_teacher.zip -O distillation_sample_teacher.zip
!unzip -q -o distillation_sample_teacher.zip -d .

from autogluon.multimodal import MultiModalPredictor

teacher_predictor = MultiModalPredictor.load("ag_distillation_sample_teacher/")
```

## Distill to Student Model

Distill knowledge into a smaller **BERT (6-layer)** student model by passing `teacher_predictor` to `.fit()`:

```python
student_predictor = MultiModalPredictor(label="label")
student_predictor.fit(
    train_df,
    tuning_data=valid_df,
    teacher_predictor=teacher_predictor,  # Enable distillation
    hyperparameters={
        "model.hf_text.checkpoint_name": "google/bert_uncased_L-6_H-768_A-12",  # 6-layer student
        "optim.max_epochs": 2,
    }
)

# Evaluate student
print(student_predictor.evaluate(data=test_df))
```

**Key Benefit**: Student trained via distillation typically outperforms direct finetuning by matching teacher's predictions/feature maps.

## Additional Resources

- **Advanced customization**: [AutoMM Distillation Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm/distillation)
- **Multilingual example**: [PAWS-X distillation](https://github.com/autogluon/autogluon/tree/master/examples/automm/distillation/automm_distillation_pawsx.py)
- **General examples**: [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm)
- **Customization guide**: [Customize AutoMM](customization.ipynb)