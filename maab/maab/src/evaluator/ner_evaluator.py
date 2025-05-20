import json
from typing import List
import numpy as np
from seqeval.metrics import f1_score, accuracy_score

from .utils import load_metadata
from .base_evaluator import BaseEvaluator


class BaseNEREvaluator(BaseEvaluator):
    """Base class for NER evaluation metrics"""
    
    def load_ner_data(self, file_path: str) -> List[List[str]]:
        """
        Load NER data from JSONL file
        
        Args:
            file_path: Path to the JSONL file containing NER data
            
        Returns:
            List[List[str]]: List of NER tag sequences
        """
        try:
            tags = []
            with open(file_path, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    tags.append(data['ner_tags'])
            return tags
            
        except Exception as e:
            raise ValueError(f"Error loading NER file {file_path}: {str(e)}")

    def calculate_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """This method should be implemented by specific NER metric classes"""
        raise NotImplementedError

    def evaluate(
        self,
        pred_path: str,
        gt_path: str,
        results_path: str,
        metadata_path: str,
        agent_name: str,
    ):
        """
        Evaluate NER predictions against ground truth
        
        Args:
            pred_path: Path to predictions JSONL file
            gt_path: Path to ground truth JSONL file
            results_path: Path to save results
            metadata_path: Path to metadata JSON
            agent_name: Name of the agent/model being evaluated
        """
        try:
            metadata = load_metadata(metadata_path)
            
            # Load predictions and ground truth
            y_pred = self.load_ner_data(pred_path)
            y_true = self.load_ner_data(gt_path)

            if len(y_pred) != len(y_true):
                raise ValueError(
                    f"Prediction and ground truth files have different number of sequences: "
                    f"pred={len(y_pred)}, true={len(y_true)}"
                )

            score = self.calculate_metric(y_true, y_pred)
            
            print(f"Evaluation Score ({self.name}): {score:.4f}")
            self.write_results(results_path, metadata, score, agent_name)

        except Exception as e:
            print(f"An error occurred during NER evaluation: {str(e)}")
            raise


class F1NEREvaluator(BaseNEREvaluator):
    """NER evaluator using F1 score metric"""
    
    @property
    def name(self) -> str:
        return "f1_ner"
    
    def calculate_metric(self, y_true: List[List[str]], y_pred: List[List[str]]) -> float:
        """
        Calculate F1 score for NER tags
        
        Args:
            y_true: List of ground truth NER tag sequences
            y_pred: List of predicted NER tag sequences
            
        Returns:
            float: F1 score
        """
        return f1_score(y_true, y_pred)


class AccuracyNEREvaluator(BaseNEREvaluator):
    """NER evaluator using accuracy score metric"""
    
    @property
    def name(self) -> str:
        return "accuracy_ner"
    
    def calculate_metric(self, y_true: List[List[str]], y_pred: List[List[str]]) -> float:
        """
        Calculate accuracy score for NER tags
        
        Args:
            y_true: List of ground truth NER tag sequences
            y_pred: List of predicted NER tag sequences
            
        Returns:
            float: Accuracy score
        """
        return accuracy_score(y_true, y_pred)
