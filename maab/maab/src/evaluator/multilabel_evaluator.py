import os
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import hamming_loss

from .base_evaluator import BaseEvaluator
from .utils import load_metadata


class MultiLabelBaseEvaluator(BaseEvaluator):
    """Base class for multi-label evaluation metrics"""

    def load_multilabel_data(self, file_path: str, label_columns: List[str]) -> np.ndarray:
        """
        Load multi-label data from different file formats

        Args:
            file_path: Path to the input file
            label_columns: List of column names containing labels

        Returns:
            np.ndarray: Array of shape (n_samples, n_labels)
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
                        if all(col in df.columns for col in label_columns):
                            break
                    except:
                        continue

                if df is None:
                    raise ValueError(f"Could not parse multi-label data from {file_path}")
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")

            # Verify all label columns exist
            missing_cols = [col for col in label_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing label columns in {file_path}: {missing_cols}")

            # Extract and stack label columns
            labels_array = df[label_columns].values
            return labels_array

        except Exception as e:
            raise ValueError(f"Error loading multi-label file {file_path}: {str(e)}")

    def evaluate(
        self,
        pred_path: str,
        gt_path: str,
        results_path: str,
        metadata_path: str,
        agent_name: str,
    ):
        """
        Evaluate multi-label predictions against ground truth

        Args:
            pred_path: Path to predictions file
            gt_path: Path to ground truth file
            results_path: Path to save results
            metadata_path: Path to metadata JSON
            agent_name: Name of the agent/model being evaluated
        """
        try:
            metadata = load_metadata(metadata_path)

            # Verify metadata contains required fields
            if not isinstance(metadata.get("label_columns"), list):
                raise ValueError("Metadata 'label_columns' must be a list for multi-label evaluation")

            label_columns = metadata["label_columns"]

            # Load multi-label data
            y_pred = self.load_multilabel_data(pred_path, label_columns)
            y_true = self.load_multilabel_data(gt_path, label_columns)

            if y_pred.shape != y_true.shape:
                raise ValueError(
                    f"Prediction and ground truth arrays have different shapes: "
                    f"pred={y_pred.shape}, true={y_true.shape}"
                )

            score = self.calculate_metric(y_true, y_pred)

            print(f"Evaluation Score ({self.name}): {score:.4f}")
            self.write_results(results_path, metadata, score, agent_name)

        except Exception as e:
            print(f"An error occurred during multi-label evaluation: {str(e)}")
            raise


class HammingLossMultiLabelEvaluator(MultiLabelBaseEvaluator):
    """Evaluator using Hamming Loss metric for multi-label classification"""

    @property
    def name(self) -> str:
        return "hamming_loss"

    def calculate_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Hamming Loss between true and predicted labels

        Args:
            y_true: Ground truth labels (n_samples, n_labels)
            y_pred: Predicted labels (n_samples, n_labels)

        Returns:
            float: Hamming Loss score
        """
        return hamming_loss(y_true, y_pred)


class AverageAccuracyMultiLabelEvaluator(MultiLabelBaseEvaluator):
    """Evaluator using Average Accuracy metric for multi-label classification"""

    @property
    def name(self) -> str:
        return "average_accuracy"

    def calculate_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Average Accuracy between true and predicted labels

        Args:
            y_true: Ground truth labels (n_samples, n_labels)
            y_pred: Predicted labels (n_samples, n_labels)

        Returns:
            float: Average Accuracy score (mean of accuracy for each label)
        """
        # Calculate equality matrix (True where predictions match ground truth)
        correct_predictions = y_true == y_pred

        # Calculate accuracy for each label (column-wise mean)
        label_accuracies = np.mean(correct_predictions, axis=0)

        # Return the average accuracy across all labels
        return np.mean(label_accuracies)
