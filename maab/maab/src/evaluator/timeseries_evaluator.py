import gzip
import json
import os
from typing import Any, Dict

import autogluon.timeseries.metrics as ts_metrics
import numpy as np
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame

from .base_evaluator import BaseEvaluator


def load_jsonl_gz(file_path, freq):
    """Load compressed JSONL file and convert to DataFrame."""
    data_list = []
    with gzip.open(file_path, "rt") as f:
        for line in f:
            data_list.append(json.loads(line))

    # Process each record
    processed_data = []
    for record in data_list:
        item_id = record["item_id"]
        start_time = pd.to_datetime(record["start"])

        # Create timestamps for each target value
        timestamps = pd.date_range(start=start_time, periods=len(record["target"]), freq=freq)

        # Create individual records
        for ts, target in zip(timestamps, record["target"]):
            processed_data.append({"item_id": item_id, "timestamp": ts, "target": target})

    return pd.DataFrame(processed_data)


def load_jsonl(file_path, freq):
    """Load uncompressed JSONL file and convert to DataFrame."""
    data_list = []
    with open(file_path, "r") as f:
        for line in f:
            data_list.append(json.loads(line))

    # Process each record
    processed_data = []
    for record in data_list:
        item_id = record["item_id"]
        start_time = pd.to_datetime(record["start"])

        # Create timestamps for each target value
        timestamps = pd.date_range(start=start_time, periods=len(record["target"]), freq=freq)

        # Create individual records
        for ts, target in zip(timestamps, record["target"]):
            processed_data.append({"item_id": item_id, "timestamp": ts, "target": target})

    return pd.DataFrame(processed_data)


class BaseTimeseriesEvaluator(BaseEvaluator):
    """Base class for timeseries evaluation metrics"""

    def __init__(self, metric_name: str):
        """
        Initialize the timeseries evaluator

        Args:
            metric_name: Name of the metric to use (must be available in autogluon.timeseries.metrics)
        """
        self.metric_name = metric_name

        # Validate metric exists in autogluon.timeseries.metrics
        if not hasattr(ts_metrics, metric_name):
            raise ValueError(f"Metric '{metric_name}' not found in autogluon.timeseries.metrics")
        self.metric_func = getattr(ts_metrics, metric_name)()

    @property
    def name(self) -> str:
        """Return the name of the metric being used"""
        return self.metric_name.lower()

    def calculate_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        This method is overridden but not used in timeseries evaluation.
        The actual calculation happens in evaluate() using TimeSeriesDataFrame.
        """
        raise NotImplementedError("For timeseries evaluation, use evaluate() method directly with file paths")

    def evaluate(
        self,
        pred_path: str,
        gt_path: str,
        results_path: str,
        metadata_path: str,
        agent_name: str,
    ):
        """
        Evaluate timeseries predictions against ground truth

        Args:
            pred_path: Path to predictions file
            gt_path: Path to ground truth file
            results_path: Path to save results
            metadata_path: Path to metadata JSON
            agent_name: Name of the agent/model being evaluated
        """
        try:
            metadata = self.load_metadata(metadata_path)
            prediction_length = metadata.get("prediction_length")
            id_column = metadata.get("id_column")
            timestamp_column = metadata.get("timestamp_column")
            target_column = metadata.get("label_column")
            freq = metadata.get("freq")

            if not all([prediction_length, id_column, timestamp_column, target_column]):
                raise ValueError("Metadata must contain 'id_column', 'timestamp_column', and 'label_column'")

            # Load ground truth data
            groundtruth_df = self._load_file(gt_path, freq)
            groundtruth_data = TimeSeriesDataFrame.from_data_frame(
                groundtruth_df, id_column=id_column, timestamp_column=timestamp_column
            )

            # Load predictions
            predicted_df = self._load_file(pred_path, freq)
            predicted_df["mean"] = predicted_df[target_column]
            predicted_data = TimeSeriesDataFrame.from_data_frame(
                predicted_df, id_column=id_column, timestamp_column=timestamp_column
            )

            try:
                # Calculate metric
                score = self.metric_func(
                    data=groundtruth_data,
                    predictions=predicted_data,
                    prediction_length=prediction_length,
                    target=target_column,
                )
            except Exception as e:
                print(groundtruth_data.index)
                print(predicted_data.index)
                raise e

            print(f"Evaluation Score ({self.name}): {score:.4f}")
            self.write_results(results_path, metadata, score, agent_name)

        except Exception as e:
            print(f"An error occurred during timeseries evaluation: {str(e)}")
            raise

    def _load_file(self, file_path, freq):
        """Load data file based on extension."""
        if file_path.endswith(".jsonl.gz") or file_path.endswith(".json.gz"):
            return load_jsonl_gz(file_path, freq)
        elif file_path.endswith(".jsonl") or file_path.endswith(".json"):
            return load_jsonl(file_path, freq)
        elif file_path.endswith(".csv"):
            return pd.read_csv(file_path)
        elif file_path.endswith(".csv.gz"):
            return pd.read_csv(file_path, compression="gzip")
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

    @staticmethod
    def load_metadata(metadata_path: str) -> Dict[str, Any]:
        """Load metadata from JSON file"""
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path, "r") as f:
            return json.load(f)


class SQLTimeseriesEvaluator(BaseTimeseriesEvaluator):
    def __init__(self):
        super().__init__("SQL")


class WQLTimeseriesEvaluator(BaseTimeseriesEvaluator):
    def __init__(self):
        super().__init__("WQL")


class MAETimeseriesEvaluator(BaseTimeseriesEvaluator):
    def __init__(self):
        super().__init__("MAE")


class MASETimeseriesEvaluator(BaseTimeseriesEvaluator):
    def __init__(self):
        super().__init__("MASE")


class WAPETimeseriesEvaluator(BaseTimeseriesEvaluator):
    def __init__(self):
        super().__init__("WAPE")


class MSETimeseriesEvaluator(BaseTimeseriesEvaluator):
    def __init__(self):
        super().__init__("MSE")


class RMSETimeseriesEvaluator(BaseTimeseriesEvaluator):
    def __init__(self):
        super().__init__("RMSE")


class RMSLETimeseriesEvaluator(BaseTimeseriesEvaluator):
    def __init__(self):
        super().__init__("RMSLE")


class RMSSETimeseriesEvaluator(BaseTimeseriesEvaluator):
    def __init__(self):
        super().__init__("RMSSE")


class MAPETimeseriesEvaluator(BaseTimeseriesEvaluator):
    def __init__(self):
        super().__init__("MAPE")


class SMAPETimeseriesEvaluator(BaseTimeseriesEvaluator):
    def __init__(self):
        super().__init__("SMAPE")
