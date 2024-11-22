"""Predictors solve tabular prediction tasks"""

import logging
from collections import defaultdict
from typing import Any, Dict, Optional

import numpy as np
from autogluon.common.features.feature_metadata import FeatureMetadata
from autogluon.core.metrics import make_scorer
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.metrics import mean_squared_log_error

from .constants import BINARY, CLASSIFICATION_PROBA_EVAL_METRIC, MULTICLASS
from .task import TabularPredictionTask

logger = logging.getLogger(__name__)


def rmsle_func(y_true, y_pred, **kwargs):
    y_pred = np.clip(y_pred, a_min=0.0, a_max=None)
    return np.sqrt(mean_squared_log_error(y_true, y_pred, **kwargs))


root_mean_square_logarithmic_error = make_scorer(
    "root_mean_square_logarithmic_error",
    rmsle_func,
    optimum=0,
    greater_is_better=False,
)


class Predictor:
    def fit(self, task: TabularPredictionTask, time_limit: Optional[float] = None) -> "Predictor":
        return self

    def predict(self, task: TabularPredictionTask) -> Any:
        raise NotImplementedError

    def fit_predict(self, task: TabularPredictionTask) -> Any:
        return self.fit(task).predict(task)


class AutogluonTabularPredictor(Predictor):
    def __init__(self, config: Any):
        self.config = config
        self.metadata: Dict[str, Any] = defaultdict(dict)
        self.tabular_predictor: TabularPredictor = None

    def save_dataset_details(self, task: TabularPredictionTask) -> None:
        for key, data in (
            ("train", task.train_data),
            ("test", task.test_data),
        ):
            self.metadata["dataset_summary"][key] = data.describe().to_dict()
            self.metadata["feature_metadata_raw"][key] = FeatureMetadata.from_df(data).to_dict()
            self.metadata["feature_missing_values"][key] = (data.isna().sum() / len(data)).to_dict()

    def describe(self) -> Dict[str, Any]:
        return dict(self.metadata)

    def fit(self, task: TabularPredictionTask, time_limit: Optional[float] = None) -> "AutogluonTabularPredictor":
        """Trains an AutoGluon TabularPredictor with parsed arguments. Saves trained predictor to
        `self.predictor`.

        Raises
        ------
        Exception
            TabularPredictor fit failures
        """
        eval_metric = task.eval_metric
        if eval_metric == "root_mean_squared_logarithmic_error":
            eval_metric = root_mean_square_logarithmic_error

        predictor_init_kwargs = {
            "learner_kwargs": {"ignored_columns": task.columns_in_train_but_not_test},
            "label": task.label_column,
            "problem_type": task.problem_type,
            "eval_metric": eval_metric,
            **self.config.predictor_init_kwargs,
        }
        predictor_fit_kwargs = self.config.predictor_fit_kwargs.copy()
        predictor_fit_kwargs.pop("time_limit", None)

        logger.info("Fitting AutoGluon TabularPredictor")
        logger.info(f"predictor_init_kwargs: {predictor_init_kwargs}")
        logger.info(f"predictor_fit_kwargs: {predictor_fit_kwargs}")

        self.metadata |= {
            "predictor_init_kwargs": predictor_init_kwargs,
            "predictor_fit_kwargs": predictor_fit_kwargs,
        }
        self.save_dataset_details(task)
        self.predictor = TabularPredictor(**predictor_init_kwargs).fit(
            task.train_data, **predictor_fit_kwargs, time_limit=time_limit
        )

        self.metadata["leaderboard"] = self.predictor.leaderboard().to_dict()
        return self

    def predict(self, task: TabularPredictionTask) -> TabularDataset:
        """Calls `TabularPredictor.predict` or `TabularPredictor.predict_proba` on `self.transformed_test_data`.
        Saves predictions to `self.predictions`.

        Raises
        ------
        Exception
            `TabularPredictor.predict` fails
        """
        if task.eval_metric in CLASSIFICATION_PROBA_EVAL_METRIC and self.predictor.problem_type in [
            BINARY,
            MULTICLASS,
        ]:
            return self.predictor.predict_proba(
                task.test_data, as_multiclass=(self.predictor.problem_type == MULTICLASS)
            )
        else:
            return self.predictor.predict(task.test_data)
