Summary: This tutorial demonstrates knowledge distillation in AutoGluon MultiModal, showing how to transfer knowledge from a larger teacher model to a smaller student model. It covers implementation techniques for loading pre-trained models, configuring distillation, and evaluating performance. The code helps with creating efficient, smaller models that maintain accuracy by learning from larger ones. Key features include: simple distillation activation via the teacher_predictor parameter, working with BERT models of different sizes (12-layer to 6-layer), dataset preparation for NLP tasks, and model evaluation. The implementation focuses on practical application with minimal configuration requirements.

```python
!pip install autogluon.multimodal

```


```python
import datasets
from datasets import load_dataset

datasets.logging.disable_progress_bar()

dataset = load_dataset("glue", "qnli")
```


```python
dataset['train']
```


```python
from sklearn.model_selection import train_test_split

train_valid_df = dataset["train"].to_pandas()[["question", "sentence", "label"]].sample(1000, random_state=123)
train_df, valid_df = train_test_split(train_valid_df, test_size=0.2, random_state=123)
test_df = dataset["validation"].to_pandas()[["question", "sentence", "label"]].sample(1000, random_state=123)
```

## Load the Teacher Model

In our example, we will directly load a teacher model with the [google/bert_uncased_L-12_H-768_A-12](https://huggingface.co/google/bert_uncased_L-12_H-768_A-12) backbone that has been trained on QNLI and distill it into a student model with the [google/bert_uncased_L-6_H-768_A-12](https://huggingface.co/google/bert_uncased_L-6_H-768_A-12) backbone.


```python
!wget --quiet https://automl-mm-bench.s3.amazonaws.com/unit-tests/distillation_sample_teacher.zip -O distillation_sample_teacher.zip
!unzip -q -o distillation_sample_teacher.zip -d .
```


```python
from autogluon.multimodal import MultiModalPredictor

teacher_predictor = MultiModalPredictor.load("ag_distillation_sample_teacher/")
```

## Distill to Student

Training the student model is straight forward. You may just add the `teacher_predictor` argument when calling `.fit()`. 
Internally, the student will be trained by matching the prediction/feature map from the teacher. It can perform better than 
directly finetuning the student.


```python
student_predictor = MultiModalPredictor(label="label")
student_predictor.fit(
    train_df,
    tuning_data=valid_df,
    teacher_predictor=teacher_predictor,
    hyperparameters={
        "model.hf_text.checkpoint_name": "google/bert_uncased_L-6_H-768_A-12",
        "optim.max_epochs": 2,
    }
)
```


```python
print(student_predictor.evaluate(data=test_df))
```

## More about Knowledge Distillation

To learn how to customize distillation and how it compares with direct finetuning, see the distillation examples 
and README in [AutoMM Distillation Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm/distillation).
Especially the [multilingual distillation example](https://github.com/autogluon/autogluon/tree/master/examples/automm/distillation/automm_distillation_pawsx.py) with more details and customization.

## Other Examples

You may go to [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm) to explore other examples about AutoMM.

## Customization
To learn how to customize AutoMM, please refer to [Customize AutoMM](customization.ipynb).
