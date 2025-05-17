import pandas as pd
import numpy as np
import os
from typing import Dict, Any
from abc import ABC, abstractmethod

from .utils import load_metadata


class BaseEvaluator(ABC):
    @abstractmethod
    def calculate_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    def load_data(self, file_path: str, label_column: str) -> np.ndarray:
        """
        Load data from different file formats (CSV, Parquet, TXT)
        
        Args:
            file_path: Path to the input file
            label_column: Name of the column containing labels/values
            
        Returns:
            np.ndarray: Array of values from the specified column
        """
        file_extension = os.path.splitext(file_path)[1].lower()

        try:
            if file_extension == ".csv":
                df = pd.read_csv(file_path)
            elif file_extension == ".parquet" or file_extension == ".pq":
                df = pd.read_parquet(file_path)
            elif file_extension == ".txt":
                # Try different delimiters for txt files
                delimiters = [",", "\t", " ", "|"]
                df = None
                for delimiter in delimiters:
                    try:
                        df = pd.read_csv(file_path, delimiter=delimiter)
                        # Check if we got more than one column and the result makes sense
                        if len(df.columns) > 1 or (
                            len(df.columns) == 1 and label_column in df.columns
                        ):
                            break
                    except:
                        continue

                if df is None:
                    # If no delimiter worked, try reading as single column
                    try:
                        values = np.loadtxt(file_path)
                        return values
                    except:
                        # Last resort: read raw lines and try to convert to numbers
                        with open(file_path, "r") as f:
                            lines = f.readlines()
                        values = []
                        for line in lines:
                            try:
                                value = float(line.strip())
                                values.append(value)
                            except ValueError:
                                continue
                        if not values:
                            raise ValueError(f"Could not parse values from {file_path}")
                        return np.array(values)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")

            # For DataFrame cases, handle the label column
            if df is not None:
                if label_column not in df.columns:
                    # If label_column not found, try using the first column
                    if len(df.columns) == 1:
                        return df.iloc[:, 0].values
                    else:
                        raise ValueError(
                            f"'{label_column}' column not found in {file_path} and multiple columns present"
                        )
                return df[label_column].values

        except Exception as e:
            raise ValueError(f"Error loading file {file_path}: {str(e)}")

    def write_results(
        self, results_path: str, metadata: Dict[str, Any], score: float, agent_name: str
    ):
        """Write evaluation results to a file (CSV format)"""
        results_df = pd.DataFrame()
        if os.path.exists(results_path):
            results_df = pd.read_csv(results_path)

        new_row = {**metadata, "result": score, "agent_name": agent_name}
        new_row_df = pd.DataFrame([new_row])
        results_df = pd.concat([results_df, new_row_df], ignore_index=True)
        results_df.to_csv(results_path, index=False)
        print(f"Results written to {results_path}")

    def evaluate(
        self,
        pred_path: str,
        gt_path: str,
        results_path: str,
        metadata_path: str,
        agent_name: str,
    ):
        """
        Evaluate predictions against ground truth using the specified metric
        
        Args:
            pred_path: Path to predictions file
            gt_path: Path to ground truth file
            results_path: Path to save results
            metadata_path: Path to metadata JSON
            agent_name: Name of the agent/model being evaluated
        """
        try:
            metadata = load_metadata(metadata_path)
            label_column = metadata["label_column"]

            # Special handling for metrics that need raw file paths
            if self.name in ["map", "iou"]:
                score = self.calculate_metric(gt_path, pred_path)
            else:
                y_pred = self.load_data(pred_path, label_column)
                y_true = self.load_data(gt_path, label_column)

                if len(y_pred) != len(y_true):
                    raise ValueError(
                        f"Prediction and ground truth files have different lengths: "
                        f"pred={len(y_pred)}, true={len(y_true)}"
                    )

                score = self.calculate_metric(y_true, y_pred)

            print(f"Evaluation Score ({self.name}): {score:.4f}")
            self.write_results(results_path, metadata, score, agent_name)

        except Exception as e:
            print(f"An error occurred during evaluation: {str(e)}")
            raise
