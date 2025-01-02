from pathlib import Path

import pandas as pd
import pytest

from autogluon.assistant.constants import TEST, TRAIN
from autogluon.assistant.task import TabularPredictionTask


@pytest.fixture
def test_data():
    data = pd.DataFrame({"id": [1, 2, 3], "feature1": [10, 20, 30], "feature2": ["a", "b", "c"], "target": [0, 1, 0]})

    xlsx_path = Path("test.xlsx")
    xls_path = Path("test.xls")

    try:
        data.to_excel(xlsx_path, index=False, engine="openpyxl")
        data.to_excel(xls_path, index=False, engine="openpyxl")
    except Exception as e:
        xlsx_path.unlink(missing_ok=True)
        xls_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to create test files: {str(e)}")

    yield data, xlsx_path, xls_path

    xlsx_path.unlink(missing_ok=True)
    xls_path.unlink(missing_ok=True)


def test_excel_loading(test_data):
    data, xlsx_path, xls_path = test_data

    task = TabularPredictionTask(
        name="excel-test",
        description="Test excel loading functionality",
        filepaths=[xlsx_path, xls_path],
        metadata={},
        cache_data=True,
    )

    # Test XLSX loading
    task.dataset_mapping[TRAIN] = xlsx_path
    original_df = data.copy()
    original_df.reset_index(drop=True, inplace=True)

    loaded_df = task.train_data
    loaded_df.reset_index(drop=True, inplace=True)
    pd.testing.assert_frame_equal(loaded_df, original_df, check_dtype=False)

    # Test XLS loading
    task.dataset_mapping[TEST] = xls_path
    loaded_df = task.test_data
    loaded_df.reset_index(drop=True, inplace=True)
    pd.testing.assert_frame_equal(loaded_df, original_df, check_dtype=False)


def test_excel_loading_without_caching(test_data):
    _, xlsx_path, xls_path = test_data

    task = TabularPredictionTask(
        name="excel-test",
        description="Test excel loading functionality",
        filepaths=[xlsx_path, xls_path],
        metadata={},
        cache_data=False,
    )

    task.dataset_mapping[TRAIN] = xlsx_path
    assert isinstance(task.dataset_mapping[TRAIN], Path)

    task.dataset_mapping[TEST] = xls_path
    assert isinstance(task.dataset_mapping[TEST], Path)


if __name__ == "__main__":
    pytest.main([__file__])
