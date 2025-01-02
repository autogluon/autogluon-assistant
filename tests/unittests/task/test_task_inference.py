from pathlib import Path

import pandas as pd
import pytest
from hydra import compose, initialize

from autogluon.assistant.constants import OUTPUT, TEST, TRAIN
from autogluon.assistant.llm import LLMFactory
from autogluon.assistant.task import TabularPredictionTask
from autogluon.assistant.task_inference.task_inference import LabelColumnInference

_config_path = "../../../src/autogluon/assistant/configs"
with initialize(version_base=None, config_path=_config_path):
    config = compose(config_name="default")
_llm = LLMFactory.get_chat_model(config.llm)


@pytest.fixture
def toy_multiclass_data():
    # toy data below is formed from shelter_animal_data
    train_data = pd.DataFrame(
        {
            "AnimalID": ["A671945", "A656520", "A686464", "A683430", "A667013"],
            "Name": ["Hambone", "Emily", "Pearce", None, None],
            "DateTime": [
                "2014-02-12 18:22:00",
                "2013-10-13 12:44:00",
                "2015-01-31 12:28:00",
                "2014-07-11 19:09:00",
                "2013-11-15 12:52:00",
            ],
            "OutcomeType": ["Return_to_owner", "Euthanasia", "Adoption", "Transfer", "Died"],
            "OutcomeSubtype": [None, "Suffering", "Foster", "Partner", "Partner"],
            "AnimalType": ["Dog", "Cat", "Dog", "Cat", "Dog"],
        }
    )

    test_data = pd.DataFrame(
        {
            "ID": [1, 2, 3, 4, 5],
            "Name": ["Summer", "Cheyenne", "Gus", "Pongo", "Skooter"],
            "DateTime": [
                "2015-10-12 12:15:00",
                "2014-07-26 17:59:00",
                "2016-01-13 12:20:00",
                "2013-12-28 18:12:00",
                "2015-09-24 17:59:00",
            ],
            "AnimalType": ["Dog", "Dog", "Cat", "Dog", "Dog"],
        }
    )

    sample_submission_data = pd.DataFrame(
        {
            "ID": [1, 2, 3, 4, 5],
            "Adoption": [1, 1, 1, 1, 1],
            "Died": [0, 0, 0, 0, 0],
            "Euthanasia": [0, 0, 0, 0, 0],
            "Return_to_owner": [0, 0, 0, 0, 0],
            "Transfer": [0, 0, 0, 0, 0],
        }
    )

    return train_data, test_data, sample_submission_data


def test_label_column_inference(toy_multiclass_data):
    train, test, sample_submission = toy_multiclass_data

    metadata = {
        "output_columns": list(sample_submission.columns),
        "label_column": None,  # Label column should be inferred
    }

    task = TabularPredictionTask(
        name="shelter-animal-outcomes",
        description="Predict the outcome type for shelter animals",
        filepaths=[Path("train.csv"), Path("test.csv"), Path("sample_submission.csv")],
        metadata=metadata,
        cache_data=False,
    )

    task.dataset_mapping[TRAIN] = train
    task.dataset_mapping[TEST] = test
    task.dataset_mapping[OUTPUT] = sample_submission

    # this won't guarantee a test for fallback logic since LLM will work most probably
    transformer = LabelColumnInference(_llm)
    task = transformer.transform(task)
    assert task.metadata["label_column"] == "OutcomeType", "The label column should be 'OutcomeType'."

    # guaranteed test fallback logic explicitly
    task.metadata["label_column"] = task._infer_label_column_from_sample_submission_data()
    assert task.metadata["label_column"] == "OutcomeType", "The label column should be 'OutcomeType'."


if __name__ == "__main__":
    pytest.main()
