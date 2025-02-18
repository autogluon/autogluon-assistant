import logging
import os
import warnings
from typing import Mapping, Tuple

import pandas as pd

from .base import BaseFeatureTransformer

logger = logging.getLogger(__name__)
warnings.filterwarnings(action="ignore")

try:
    from caafe import CAAFEClassifier
    from caafe.run_llm_code import run_llm_code
except ImportError:
    raise ImportError(
        "CAAFE required for feature generation but not installed. Please install with: `pip install caafe@git+https://github.com/anirudhdagar/CAAFE.git@main`"
    )


class CAAFETransformer(BaseFeatureTransformer):
    identifier = "caafe"

    def __init__(
        self,
        llm_provider: str = "openai",
        llm_model: str = "gpt-3.5-turbo",
        num_iterations: int = 2,
        optimization_metric: str = "roc",
        eval_model: str = "lightgbm",
        region_name: str = os.environ.get("AWS_DEFAULT_REGION", "us-west-2"),
        **kwargs,
    ) -> None:
        # Set up credentials if using OpenAI
        if llm_provider == "openai":
            import openai

            openai.api_key = kwargs.get("openai_api_key", os.environ.get("OPENAI_API_KEY"))

        pd.set_option("future.no_silent_downcasting", True)

        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.iterations = num_iterations
        self.optimization_metric = optimization_metric
        self.eval_model = eval_model
        self.region_name = region_name

        # Initialize the base classifier
        if self.eval_model == "tab_pfn":
            from tabpfn import TabPFNClassifier

            clf_no_feat_eng = TabPFNClassifier(device="cpu", N_ensemble_configurations=16)
        elif self.eval_model == "lightgbm":
            from lightgbm import LGBMClassifier

            clf_no_feat_eng = LGBMClassifier()
        else:
            raise ValueError(f"Unsupported CAAFE eval model: {self.eval_model}")

        self.caafe_clf = CAAFEClassifier(
            base_classifier=clf_no_feat_eng,
            optimization_metric=self.optimization_metric,
            llm_provider=self.llm_provider,
            llm_model=self.llm_model,
            iterations=self.iterations,
            region_name=self.region_name,
            display_method="print",
            **kwargs,  # Pass through any additional provider-specific kwargs
        )

        self.metadata = {"transformer": "CAAFE", "llm_provider": llm_provider, "llm_model": llm_model}

    def _fit_dataframes(
        self,
        train_X: pd.DataFrame,
        train_y: pd.Series,
        *,
        target_column_name: str,
        problem_type: str = "classification",
        dataset_description: str = "",
        **kwargs,
    ) -> None:
        if problem_type not in ("binary", "multiclass"):
            logger.info("Feature transformer CAAFE only supports classification problems.")
            return

        categorical_target = not pd.api.types.is_numeric_dtype(train_y)
        if categorical_target:
            encoded_y, _ = train_y.factorize()

        self.caafe_clf.fit(
            train_X.to_numpy(),
            encoded_y if categorical_target else train_y.to_numpy(),
            dataset_description,
            train_X.columns,
            target_column_name,
        )

        logger.info(f"CAAFE generated features using {self.llm_provider} model {self.llm_model}:")
        logger.info(self.caafe_clf.code)

    def _transform_dataframes(self, train_X: pd.DataFrame, test_X: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        transformed_train_X = run_llm_code(self.caafe_clf.code, train_X)
        transformed_test_X = run_llm_code(self.caafe_clf.code, test_X)

        return transformed_train_X, transformed_test_X

    def get_metadata(self) -> Mapping:
        return self.metadata
