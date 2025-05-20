import json
from abc import abstractmethod
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

from .base_evaluator import BaseEvaluator  # Import BaseEvaluator


class BaseRetrievalEvaluator(BaseEvaluator):
    """Base class for retrieval evaluation metrics"""

    def __init__(self, k: int = 10):
        """
        Initialize the retrieval evaluator

        Args:
            k: Cut-off for top-k evaluation
        """
        self.k = k

    def load_retrieval_data(self, file_path: str, metadata: Dict[str, Any]) -> pd.DataFrame:
        """Load retrieval data from file"""
        required_cols = [metadata["query_column"], metadata["corpus_column"], metadata["label_column"]]

        df = pd.read_csv(file_path, sep="\t")
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {file_path}: {missing}")

        return df

    def prepare_evaluation_data(self, pred_df: pd.DataFrame, gt_df: pd.DataFrame, metadata: Dict[str, Any]) -> tuple:
        """Prepare data for evaluation"""
        query_col = metadata["query_column"]
        doc_col = metadata["corpus_column"]
        score_col = metadata["label_column"]

        all_preds = []
        all_labels = []
        all_scores = []

        # Process each query
        for query_id in pred_df[query_col].unique():
            # Get predictions
            query_preds = pred_df[pred_df[query_col] == query_id]
            pred_docs = query_preds[doc_col].values[: self.k]
            pred_scores = query_preds[score_col].values[: self.k]

            # Pad if necessary
            if len(pred_docs) < self.k:
                pad_len = self.k - len(pred_docs)
                pred_docs = np.pad(pred_docs, (0, pad_len), mode="constant", constant_values="")
                pred_scores = np.pad(pred_scores, (0, pad_len), mode="constant")

            # Get ground truth
            query_gt = gt_df[gt_df[query_col] == query_id]
            relevant_docs = query_gt[query_gt[score_col] > 0][doc_col].values

            if len(relevant_docs) > 0:  # Only include queries with relevant docs
                all_preds.append(pred_docs)
                all_labels.append(relevant_docs)
                all_scores.append(pred_scores)

        return np.array(all_preds), all_labels, np.array(all_scores)

    def evaluate(self, pred_path: str, gt_path: str, results_path: str, metadata_path: str, agent_name: str) -> float:
        """
        Evaluate retrieval predictions against ground truth
        """
        try:
            # Load metadata
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            # Load data
            pred_df = self.load_retrieval_data(pred_path, metadata)
            gt_df = self.load_retrieval_data(gt_path, metadata)

            # Prepare evaluation data
            preds, labels, scores = self.prepare_evaluation_data(pred_df, gt_df, metadata)

            if len(preds) == 0:
                print("No valid queries found for evaluation")
                return 0.0

            # Calculate metric
            metric_value = self.calculate_metric(preds, labels, scores)

            # Print and save results
            print(f"Evaluation Score ({self.name}): {metric_value:.4f}")
            self.write_results(results_path, metadata, metric_value, agent_name)

            return metric_value

        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            raise

    # Override the calculate_metric from BaseEvaluator to match retrieval signature
    @abstractmethod
    def calculate_metric(self, preds: np.ndarray, labels: List[np.ndarray], scores: np.ndarray) -> float:
        """Calculate the metric value"""
        pass


class NDCGRetrievalEvaluator(BaseRetrievalEvaluator):
    """NDCG@k evaluator"""

    @property
    def name(self) -> str:
        return f"ndcg@{self.k}"

    def calculate_metric(self, preds: np.ndarray, labels: List[np.ndarray], scores: np.ndarray) -> float:
        """Calculate NDCG@k"""
        binary_relevance = []
        for pred, label in zip(preds, labels):
            rel_scores = [1 if doc in label else 0 for doc in pred]
            binary_relevance.append(rel_scores)

        binary_relevance = np.array(binary_relevance)
        return ndcg_score(binary_relevance, scores, k=self.k)


class RecallRetrievalEvaluator(BaseRetrievalEvaluator):
    """Recall@k evaluator"""

    @property
    def name(self) -> str:
        return f"recall@{self.k}"

    def calculate_metric(self, preds: np.ndarray, labels: List[np.ndarray], scores: np.ndarray) -> float:
        """Calculate Recall@k"""
        recall_sum = 0
        for pred, label in zip(preds, labels):
            retrieved_relevant = np.intersect1d(label, pred[: self.k])
            recall_sum += len(retrieved_relevant) / len(label)

        return recall_sum / len(preds)


class MRRRetrievalEvaluator(BaseRetrievalEvaluator):
    """Mean Reciprocal Rank evaluator"""

    @property
    def name(self) -> str:
        return f"mrr@{self.k}"

    def calculate_metric(self, preds: np.ndarray, labels: List[np.ndarray], scores: np.ndarray) -> float:
        """Calculate MRR@k"""
        mrr_sum = 0
        for pred, label in zip(preds, labels):
            for i, doc in enumerate(pred[: self.k], 1):
                if doc in label:
                    mrr_sum += 1 / i
                    break

        return mrr_sum / len(preds)
