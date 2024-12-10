"""A task encapsulates the data for a data science task or project. It contains descriptions, data, metadata."""

import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import pandas as pd
import s3fs
from autogluon.tabular import TabularDataset

from .constants import OUTPUT, TEST, TRAIN


class TabularPredictionTask:
    """A task contains data and metadata for a tabular machine learning task, including datasets, metadata such as
    problem type, test_id_column, etc.
    """

    def __init__(
        self,
        filepaths: List[Path],
        metadata: Dict[str, Any],
        name: Optional[str] = "",
        description: Optional[str] = "",
        cache_data: bool = True,
    ):
        self.metadata: Dict[str, Any] = {
            "name": name,
            "description": description,
            "label_column": None,
            "problem_type": None,
            "eval_metric": None,  # string, keying Autogluon Tabular metrics
            "test_id_column": None,
        }
        self.metadata.update(metadata)

        self.filepaths = filepaths
        self.cache_data = cache_data

        # TODO: each data split can have multiple files
        self.dataset_mapping: Dict[str, Union[Path, pd.DataFrame, TabularDataset]] = {
            TRAIN: None,
            TEST: None,
            OUTPUT: None,
        }

    def __repr__(self) -> str:
        return f"TabularPredictionTask(name={self.metadata['name']}, description={self.metadata['description'][:100]}, {len(self.dataset_mapping)} datasets)"

    @staticmethod
    def read_task_file(task_path: Path, filename_pattern: str, default_filename: str = "description.txt") -> str:
        """Recursively search for a file in the task path and return its contents."""
        try:
            first_path = sorted(
                list(task_path.glob(filename_pattern)),
                key=lambda x: len(x.parents),  # top level files takes precedence
            )
            if not first_path:
                return Path.read_text(task_path / default_filename)
            return Path.read_text(first_path[0])
        except (FileNotFoundError, IndexError):
            return ""

    @staticmethod
    def save_artifacts(full_save_path, predictor, train_data, test_data, sample_submission_data):
        # Prepare the dictionary with all artifacts
        artifacts = {
            "trained_model": predictor,  # AutoGluon TabularPredictor
            "train_data": train_data,  # Pandas DataFrame
            "test_data": test_data,  # Pandas DataFrame
            "out_data": sample_submission_data,  # Pandas DataFrame
        }

        # Get the directory where the AG models are saved
        ag_model_dir = predictor.predictor.path
        full_save_path_pkl_file = f"{full_save_path}/artifacts.pkl"
        if full_save_path.startswith("s3://"):
            # Using s3fs to save directly to S3
            fs = s3fs.S3FileSystem()
            with fs.open(full_save_path_pkl_file, "wb") as f:
                joblib.dump(artifacts, f)

            # Upload the ag model directory to the same S3 bucket
            s3_model_dir = f"{full_save_path}/{os.path.dirname(ag_model_dir)}/{os.path.basename(ag_model_dir)}"
            fs.put(ag_model_dir, s3_model_dir, recursive=True)
        else:
            # Local path handling
            os.makedirs(full_save_path, exist_ok=True)

            # Save the artifacts .pkl file locally
            with open(full_save_path_pkl_file, "wb") as f:
                joblib.dump(artifacts, f)

            # Copy the model directory to the same location as the .pkl file
            local_model_dir = os.path.join(full_save_path, ag_model_dir)
            shutil.copytree(ag_model_dir, local_model_dir, dirs_exist_ok=True)

    @classmethod
    def from_path(cls, task_root_dir: Path, name: Optional[str] = None) -> "TabularPredictionTask":
        # Get all filenames under task_root_dir
        task_data_filenames = []
        for root, _, files in os.walk(task_root_dir):
            for file in files:
                # Get the relative path
                relative_path = os.path.relpath(os.path.join(root, file), task_root_dir)
                task_data_filenames.append(relative_path)

        return cls(
            filepaths=[task_root_dir / fn for fn in task_data_filenames],
            metadata=dict(
                name=task_root_dir.name,
            ),
        )

    def describe(self) -> Dict[str, Any]:
        """Return a description of the task."""
        describe_task = {
            "name": self.metadata["name"],
            "description": self.metadata["description"],
            "metadata": self.metadata,
            "train_data": self.train_data.describe().to_dict(),
            "test_data": self.test_data.describe().to_dict(),
        }
        if self.sample_submission_data is not None:
            describe_task["sample_submission_data"] = self.sample_submission_data.describe().to_dict()

        return describe_task

    def get_filenames(self) -> List[str]:
        """Return all filenames for the task."""
        return [f.name for f in self.filepaths]

    def _set_task_files(self, dataset_name_mapping: Dict[str, Union[str, Path, pd.DataFrame, TabularDataset]]) -> None:
        """Set the task files for the task."""
        for k, v in dataset_name_mapping.items():
            if v is None:
                self.dataset_mapping[k] = None
            elif isinstance(v, (pd.DataFrame, TabularDataset)):
                self.dataset_mapping[k] = v
            elif isinstance(v, Path):
                if v.suffix in [".xlsx", ".xls"]:  # Check if the file is an Excel file
                    self.dataset_mapping[k] = pd.read_excel(v, engine="calamine") if self.cache_data else v
                else:
                    self.dataset_mapping[k] = TabularDataset(str(v)) if self.cache_data else v
            elif isinstance(v, str):
                filepath = next(
                    iter([path for path in self.filepaths if path.name == v]), self.filepaths[0].parent / v
                )
                if not filepath.is_file():
                    raise ValueError(f"File {v} not found in task {self.metadata['name']}")
                # TODO: integrate read_excel in AG Tabular
                if filepath.suffix in [".xlsx", ".xls"]:  # Check if the file is an Excel file
                    self.dataset_mapping[k] = (
                        pd.read_excel(filepath, engine="calamine") if self.cache_data else filepath
                    )
                else:
                    self.dataset_mapping[k] = TabularDataset(str(filepath)) if self.cache_data else filepath
            else:
                raise TypeError(f"Unsupported type for dataset_mapping: {type(v)}")

    @property
    def train_data(self) -> TabularDataset:
        return self.load_task_data(TRAIN)

    @train_data.setter
    def train_data(self, data: Union[str, Path, pd.DataFrame, TabularDataset]) -> None:
        self._set_task_files({TRAIN: data})

    @property
    def test_data(self) -> TabularDataset:
        """Return the test dataset for the task."""
        return self.load_task_data(TEST)

    @test_data.setter
    def test_data(self, data: Union[str, Path, pd.DataFrame, TabularDataset]) -> None:
        self._set_task_files({TEST: data})

    @property
    def sample_submission_data(self) -> TabularDataset:
        """Return the output dataset for the task."""
        return self.load_task_data(OUTPUT)

    @sample_submission_data.setter
    def sample_submission_data(self, data: Union[str, Path, pd.DataFrame, TabularDataset]) -> None:
        if self.sample_submission_data is not None:
            raise ValueError("Output data already set for task")
        self._set_task_files({OUTPUT: data})

    @property
    def output_columns(self) -> List[str]:
        """Return the output dataset columns for the task."""
        if self.sample_submission_data is None:
            if self.label_column:
                return [self.label_column]
            else:
                return None
        else:
            return self.sample_submission_data.columns.to_list()

    @property
    def label_column(self) -> Optional[str]:
        """Return the label column for the task."""
        if "label_column" in self.metadata and self.metadata["label_column"]:
            return self.metadata["label_column"]
        else:
            # should ideally never be called after LabelColumnInferenceTransformer has run
            # `_infer_label_column_from_sample_submission_data` is the fallback when label_column is not found
            return self._infer_label_column_from_sample_submission_data()

    @label_column.setter
    def label_column(self, label_column: str) -> None:
        self.metadata["label_column"] = label_column

    @property
    def columns_in_train_but_not_test(self) -> List[str]:
        return list(set(self.train_data.columns) - set(self.test_data.columns))

    @property
    def data_description(self) -> str:
        return self.metadata.get("data_description", self.description)

    @property
    def evaluation_description(self) -> str:
        return self.metadata.get("evaluation_description", self.description)

    @property
    def test_id_column(self) -> Optional[str]:
        return self.metadata.get("test_id_column", None)

    @test_id_column.setter
    def test_id_column(self, test_id_column: str) -> None:
        self.metadata["test_id_column"] = test_id_column

    @property
    def train_id_column(self) -> Optional[str]:
        return self.metadata.get("train_id_column", None)

    @train_id_column.setter
    def train_id_column(self, train_id_column: str) -> None:
        self.metadata["train_id_column"] = train_id_column

    @property
    def output_id_column(self) -> Optional[str]:
        return self.metadata.get(
            "output_id_column",
            self.sample_submission_data.columns[0] if self.sample_submission_data is not None else None,
        )

    @output_id_column.setter
    def output_id_column(self, output_id_column: str) -> None:
        self.metadata["output_id_column"] = output_id_column

    def _infer_label_column_from_sample_submission_data(self) -> Optional[str]:
        if self.output_columns is None:
            return None

        # Assume the first output column is the ID column and ignore it
        relevant_output_cols = self.output_columns[1:]

        # Check if any of the output columns exist in the train data
        existing_output_cols = [col for col in relevant_output_cols if col in self.train_data.columns]

        # Case 1: If there's only one output column in the train data, use it
        if len(existing_output_cols) == 1:
            return existing_output_cols[0]

        # Case 2: For example in some multiclass problems, look for a column
        #         whose unique values match or contain the output columns
        output_set = set(col.lower() for col in relevant_output_cols)
        for col in self.train_data.columns:
            unique_values = set(str(val).lower() for val in self.train_data[col].unique() if pd.notna(val))
            if output_set == unique_values or output_set.issubset(unique_values):
                return col

        # If no suitable column is found, raise an exception
        raise ValueError("Unable to infer the label column. Please specify it manually.")

    def _find_problem_type_in_description(self) -> Optional[str]:
        """Find the problem type in the task description."""
        if "regression" in self.description.lower():
            return "regression"
        elif "classification" in self.description.lower():
            return "binary"
        else:
            return None

    @property
    def problem_type(self) -> Optional[str]:
        return self.metadata["problem_type"] or self._find_problem_type_in_description()

    @problem_type.setter
    def problem_type(self, problem_type: str) -> None:
        self.metadata["problem_type"] = problem_type

    @property
    def eval_metric(self) -> Optional[str]:
        return self.metadata["eval_metric"] or (
            self.preferred_eval_metrics[self.problem_type] if self.problem_type else None
        )

    @eval_metric.setter
    def eval_metric(self, eval_metric: str) -> None:
        self.metadata["eval_metric"] = eval_metric

    def load_task_data(self, dataset_key: Union[str, str]) -> TabularDataset:
        """Load the competition file for the task."""
        if dataset_key not in self.dataset_mapping:
            raise ValueError(f"Dataset type {dataset_key} not found for task {self.name}")

        dataset = self.dataset_mapping[dataset_key]
        if dataset is None:
            return None
        if isinstance(dataset, pd.DataFrame):
            return TabularDataset(dataset)
        elif isinstance(dataset, TabularDataset):
            return dataset
        else:
            if isinstance(dataset, Path):
                filepath = dataset
            else:
                filepath = Path(dataset)

            # Check if the file is an Excel file
            if filepath.suffix in [".xlsx", ".xls"]:
                df = pd.read_excel(filepath, engine="calamine")
                return TabularDataset(df)
            elif filepath.suffix == ".json":
                raise TypeError(f"File {filepath.name} has unsupported type: json")
            else:
                return TabularDataset(str(filepath))
