import os
from abc import abstractmethod
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd

from .base_evaluator import BaseEvaluator
from .utils import load_metadata


class SemanticSegmentationEvaluator(BaseEvaluator):
    """Base class for semantic segmentation metrics"""

    def resolve_path(self, base_dir: str, relative_path: str) -> str:
        """
        Resolve a path relative to a base directory
        Args:
            base_dir: Base directory for path resolution
            relative_path: Relative path
        Returns:
            Absolute path to the mask file
        """
        return os.path.normpath(os.path.join(base_dir, relative_path))

    def load_mask(self, mask_input: Union[str, np.ndarray, List, Tuple]) -> np.ndarray:
        """
        Load mask from various input types and convert to binary format
        Args:
            mask_input: One of:
                - Path to the mask image file (str)
                - Numpy array
                - Nested list/tuple that can be converted to numpy array
        Returns:
            Binary mask as numpy array
        """
        if isinstance(mask_input, np.ndarray):
            # If input is already a numpy array, just ensure it's binary
            if mask_input.dtype == bool:
                return mask_input
            return mask_input > 0

        if isinstance(mask_input, (list, tuple)):
            # Convert list/tuple to numpy array
            try:
                mask = np.asarray(mask_input)
                return mask > 0
            except Exception as e:
                raise ValueError(f"Failed to convert input to numpy array: {str(e)}")

        if isinstance(mask_input, str):
            # Handle file path input
            if not os.path.exists(mask_input):
                raise FileNotFoundError(f"Mask file not found: {mask_input}")

            mask = cv2.imread(mask_input, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Failed to load mask: {mask_input}")

            return mask > 0

        raise TypeError(
            f"Unsupported mask input type: {type(mask_input)}. Expected string (file path) or numpy array."
        )

    @abstractmethod
    def calculate_pair_metric(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """
        Calculate metric for a single pair of masks
        Args:
            mask1: First binary mask
            mask2: Second binary mask
        Returns:
            float: Metric score for the mask pair
        """
        pass

    def calculate_metric(
        self,
        y_true: Union[List[str], List[np.ndarray], List[List], List[Tuple]],
        y_pred: Union[List[str], List[np.ndarray], List[List], List[Tuple]],
        gt_base_dir: Optional[str] = None,
        pred_dir: Optional[str] = None,
    ) -> float:
        """
        Calculate mean metric score for list of mask pairs
        Args:
            y_true: List of ground truth masks (either file paths or numpy arrays)
            y_pred: List of predicted masks (either file paths or numpy arrays)
            gt_base_dir: Base directory for ground truth paths (only needed if using file paths)
            pred_dir: Output directory for prediction paths (only needed if using file paths)
        Returns:
            float: Mean metric score across all mask pairs
        """
        if len(y_true) != len(y_pred):
            raise ValueError(
                f"Number of ground truth masks ({len(y_true)}) does not match predictions ({len(y_pred)})"
            )

        # Determine if we're working with paths or arrays
        using_paths = isinstance(y_true[0], str) and isinstance(y_pred[0], str)
        if using_paths and (gt_base_dir is None or pred_dir is None):
            raise ValueError("Base directories must be provided when using file paths")

        scores = []
        total_pairs = len(y_true)
        processed_pairs = 0

        for gt_input, pred_input in zip(y_true, y_pred):
            try:
                # Handle path resolution if working with file paths
                if using_paths:
                    gt_path = self.resolve_path(gt_base_dir, gt_input)
                    pred_path = self.resolve_path(pred_dir, pred_input)
                    gt_input = gt_path
                    pred_input = pred_path

                # Load masks (will handle both file paths and arrays)
                gt_mask = self.load_mask(gt_input)
                pred_mask = self.load_mask(pred_input)

                # Calculate metric for this pair
                score = self.calculate_pair_metric(gt_mask, pred_mask)
                scores.append(score)

                processed_pairs += 1
                if processed_pairs % 10 == 0:
                    print(f"Processed {processed_pairs}/{total_pairs} mask pairs...")

            except Exception as e:
                error_info = f"GT: {gt_input}\nPred: {pred_input}" if using_paths else "mask pair"
                import random

                if random.randint(1, 200) == 1:
                    print(f"Warning: Error processing {error_info}\nError: {str(e)}")
                continue

        if not scores:
            raise ValueError("No valid mask pairs found for evaluation")

        print(f"Successfully processed {len(scores)}/{total_pairs} mask pairs")
        return np.mean(scores)

    def evaluate(
        self,
        pred_input: Union[str, List[np.ndarray], List[List], List[Tuple]],
        gt_input: Union[str, List[np.ndarray], List[List], List[Tuple]],
        results_path: str,
        metadata_path: Optional[str] = None,
        agent_name: str = None,
    ):
        """
        Evaluate segmentation masks using the specific metric
        Args:
            pred_input: Either path to predictions CSV file or list of prediction masks as numpy arrays
            gt_input: Either path to ground truth CSV file or list of ground truth masks as numpy arrays
            results_path: Path to save results
            metadata_path: Path to metadata JSON (only needed if using CSV files)
            agent_name: Name of the agent/model being evaluated
        """
        try:
            if isinstance(pred_input, str) and isinstance(gt_input, str):
                # Handle CSV file input
                if metadata_path is None:
                    raise ValueError("metadata_path must be provided when using CSV files")

                metadata = load_metadata(metadata_path)
                label_column = metadata["label_column"]

                # Get directories
                gt_base_dir = os.path.dirname(os.path.abspath(gt_input))
                pred_dir = os.path.dirname(os.path.abspath(pred_input))

                # Load paths from CSV files
                pred_df = pd.read_csv(pred_input)
                gt_df = pd.read_csv(gt_input)

                if label_column not in pred_df.columns or label_column not in gt_df.columns:
                    raise ValueError(f"Label column '{label_column}' not found in CSV files")

                pred_masks = pred_df[label_column].tolist()
                gt_masks = gt_df[label_column].tolist()

                print(f"Processing {len(pred_df)} mask pairs...")
                print(f"Ground truth base directory: {gt_base_dir}")
                print(f"Pred directory: {pred_dir}")

                # Calculate metric with paths
                score = self.calculate_metric(gt_masks, pred_masks, gt_base_dir, pred_dir)

            else:
                # Handle direct numpy array input
                if not isinstance(pred_input, list) or not isinstance(gt_input, list):
                    raise TypeError("When not using CSV files, inputs must be lists of numpy arrays")

                print(f"Processing {len(pred_input)} mask pairs...")
                score = self.calculate_metric(gt_input, pred_input)

            print(f"Evaluation Score ({self.name}): {score:.4f}")

            if metadata_path:
                metadata = load_metadata(metadata_path)
            else:
                metadata = {"data_format": "numpy_arrays"}

            self.write_results(results_path, metadata, score, agent_name)

        except Exception as e:
            print(f"An error occurred during evaluation: {str(e)}")
            raise


class IouEvaluator(SemanticSegmentationEvaluator):
    """Evaluator for Intersection over Union metric"""

    def calculate_pair_metric(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """
        Calculate IoU between two binary masks
        Args:
            mask1: First binary mask
            mask2: Second binary mask
        Returns:
            float: IoU score between 0 and 1
        """
        if mask1 is None or mask2 is None:
            return 0.0

        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return float(intersection) / float(union) if union > 0 else 0.0

    @property
    def name(self) -> str:
        return "iou"


class SMeasureEvaluator(SemanticSegmentationEvaluator):
    """Evaluator for Structure-measure metric"""

    def calculate_region_similarity(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Calculate region-aware structural similarity"""
        # Convert masks to float for calculations
        mask1 = mask1.astype(float)
        mask2 = mask2.astype(float)

        # Calculate means
        mean1 = np.mean(mask1)
        mean2 = np.mean(mask2)

        # Calculate the similarity of foreground and background regions
        sum_mask1 = np.sum(mask1)
        sum_mask2 = np.sum(mask2)

        if sum_mask1 == 0 and sum_mask2 == 0:
            return 1.0  # Both empty masks are considered similar
        elif sum_mask1 == 0 or sum_mask2 == 0:
            return 0.0  # One empty mask is considered dissimilar

        # Region similarity
        min_mean = min(mean1, mean2)
        max_mean = max(mean1, mean2)
        if max_mean == 0:
            return 1.0

        return 1.0 - (max_mean - min_mean) / max_mean

    def calculate_object_similarity(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Calculate object-aware structural similarity"""
        h, w = mask1.shape
        if h * w == 0:
            return 0.0

        # Calculate object centers
        x_indices = np.arange(w)
        y_indices = np.arange(h)
        X, Y = np.meshgrid(x_indices, y_indices)

        # Compute centroids for both masks
        area1 = np.sum(mask1)
        area2 = np.sum(mask2)

        if area1 == 0 and area2 == 0:
            return 1.0  # Both empty masks are considered similar
        elif area1 == 0 or area2 == 0:
            return 0.0  # One empty mask is considered dissimilar

        x1_centroid = np.sum(X * mask1) / area1
        y1_centroid = np.sum(Y * mask1) / area1
        x2_centroid = np.sum(X * mask2) / area2
        y2_centroid = np.sum(Y * mask2) / area2

        # Calculate distance between centroids
        centroid_distance = np.sqrt((x1_centroid - x2_centroid) ** 2 + (y1_centroid - y2_centroid) ** 2)
        diagonal = np.sqrt(h**2 + w**2)

        # Normalize distance by diagonal length
        normalized_distance = 2.0 * centroid_distance / diagonal
        return max(0.0, 1.0 - normalized_distance)

    def calculate_pair_metric(self, mask1: np.ndarray, mask2: np.ndarray, alpha: float = 0.5) -> float:
        """
        Calculate S-measure between two binary masks
        Args:
            mask1: First binary mask
            mask2: Second binary mask
            alpha: Weight balance parameter between region and object similarity (default: 0.5)
        Returns:
            float: S-measure score between 0 and 1
        """
        # Ensure masks are binary
        mask1 = mask1.astype(bool)
        mask2 = mask2.astype(bool)

        # Calculate region-aware and object-aware structural similarities
        region_sim = self.calculate_region_similarity(mask1, mask2)
        object_sim = self.calculate_object_similarity(mask1, mask2)

        # Combine similarities using weighted sum
        s_measure = alpha * region_sim + (1 - alpha) * object_sim
        return float(s_measure)

    @property
    def name(self) -> str:
        return "s_measure"
