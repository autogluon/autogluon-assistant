# Condensed: Knowledge Distillation in AutoMM

Summary: This tutorial demonstrates knowledge distillation implementation in AutoMM, specifically focusing on transferring knowledge from large BERT teacher models to smaller student models. It covers essential techniques for model compression using the AutoGluon framework, including data preparation with QNLI dataset, teacher model setup, and student model training configuration. Key functionalities include working with MultiModalPredictor, handling train/valid/test splits, and configuring model architectures (12-layer BERT teacher to 6-layer BERT student). The tutorial is particularly useful for tasks involving model compression, deployment optimization, and maintaining performance while reducing model size through distillation techniques.

*This is a condensed version that preserves essential implementation details and context.*

Here's the condensed tutorial focusing on essential implementation details:

# Knowledge Distillation in AutoMM

## Key Concepts
- Knowledge distillation transfers knowledge from large teacher models to smaller student models
- Enables deployment of smaller models while maintaining performance benefits from larger models
- Useful for scenarios with limited deployment resources

## Implementation

### 1. Data Preparation
```python
import datasets
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# Load QNLI dataset
dataset = load_dataset("glue", "qnli")

# Prepare train/valid/test splits
train_valid_df = dataset["train"].to_pandas()[["question", "sentence", "label"]].sample(1000, random_state=123)
train_df, valid_df = train_test_split(train_valid_df, test_size=0.2, random_state=123)
test_df = dataset["validation"].to_pandas()[["question", "sentence", "label"]].sample(1000, random_state=123)
```

### 2. Teacher Model Setup
```python
from autogluon.multimodal import MultiModalPredictor

# Load pre-trained teacher model
teacher_predictor = MultiModalPredictor.load("ag_distillation_sample_teacher/")
```

### 3. Student Model Training
```python
student_predictor = MultiModalPredictor(label="label")
student_predictor.fit(
    train_df,
    tuning_data=valid_df,
    teacher_predictor=teacher_predictor,
    hyperparameters={
        "model.hf_text.checkpoint_name": "google/bert_uncased_L-6_H-768_A-12",
        "optimization.max_epochs": 2,
    }
)
```

## Important Configurations
- Teacher model: BERT (12 layers) - `google/bert_uncased_L-12_H-768_A-12`
- Student model: BERT (6 layers) - `google/bert_uncased_L-6_H-768_A-12`

## Best Practices
1. Use validation data during training to monitor performance
2. Adjust hyperparameters based on your specific use case
3. Consider model size vs. performance tradeoffs when selecting student architecture

## Additional Resources
- Detailed examples: [AutoMM Distillation Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm/distillation)
- Multilingual distillation: [PAWS-X example](https://github.com/autogluon/autogluon/tree/master/examples/automm/distillation/automm_distillation_pawsx.py)
- Customization guide: Refer to customization.ipynb