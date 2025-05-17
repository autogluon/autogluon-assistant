import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    accuracy_score,
    mean_absolute_error,
    roc_auc_score,
    median_absolute_error,
    log_loss,
    f1_score,
    r2_score,
)

from .base_evaluator import BaseEvaluator


class F1WeightedEvaluator(BaseEvaluator):
    def calculate_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return f1_score(y_true, y_pred, average="weighted")

    @property
    def name(self) -> str:
        return "f1_weighted"


class F1Evaluator(BaseEvaluator):
    def calculate_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return f1_score(y_true, y_pred)

    @property
    def name(self) -> str:
        return "f1"


class RSquaredEvaluator(BaseEvaluator):
    def calculate_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return r2_score(y_true, y_pred)

    @property
    def name(self) -> str:
        return "r_squared"


class RmseEvaluator(BaseEvaluator):
    def calculate_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.sqrt(mean_squared_error(y_true, y_pred))

    @property
    def name(self) -> str:
        return "rmse"


class AccuracyEvaluator(BaseEvaluator):
    def calculate_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return accuracy_score(y_true, y_pred)

    @property
    def name(self) -> str:
        return "accuracy"


class RmsleEvaluator(BaseEvaluator):
    def calculate_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))

    @property
    def name(self) -> str:
        return "rmsle"


class MaeEvaluator(BaseEvaluator):
    def calculate_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return mean_absolute_error(y_true, y_pred)

    @property
    def name(self) -> str:
        return "mae"


class AurocEvaluator(BaseEvaluator):
    def calculate_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return roc_auc_score(y_true, y_pred[:, 1] if y_pred.ndim > 1 else y_pred)

    @property
    def name(self) -> str:
        return "auroc"


class MedaeEvaluator(BaseEvaluator):
    def calculate_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return median_absolute_error(y_true, y_pred)

    @property
    def name(self) -> str:
        return "medae"


class NllEvaluator(BaseEvaluator):
    def calculate_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return log_loss(
            y_true, y_pred if y_pred.ndim > 1 else np.column_stack((1 - y_pred, y_pred))
        )

    @property
    def name(self) -> str:
        return "nll"
