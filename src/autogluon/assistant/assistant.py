import logging
import signal
import sys
import threading
from contextlib import contextmanager
from typing import Any, Dict, Optional, Union

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from autogluon.assistant.llm import (
    AssistantChatBedrock,
    AssistantChatOpenAI,
    AssistantAzureChatOpenAI,
    LLMFactory,
)

from .predictor import AutogluonTabularPredictor
from .task import TabularPredictionTask
from .task_inference import (
    DataFileNameInference,
    DescriptionFileNameInference,
    EvalMetricInference,
    LabelColumnInference,
    OutputIDColumnInference,
    ProblemTypeInference,
    TestIDColumnInference,
    TrainIDColumnInference,
)
from .utils import get_feature_transformers_config

logger = logging.getLogger(__name__)


@contextmanager
def timeout(seconds: int, error_message: Optional[str] = None):
    if sys.platform == "win32":
        # Windows implementation using threading
        timer = threading.Timer(
            seconds, lambda: (_ for _ in ()).throw(TimeoutError(error_message))
        )
        timer.start()
        try:
            yield
        finally:
            timer.cancel()
    else:
        # Unix implementation using SIGALRM
        def handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        signal.signal(signal.SIGALRM, handle_timeout)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)


class TabularPredictionAssistant:
    """A TabularPredictionAssistant performs a supervised tabular learning task"""

    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.llm: Union[
            AssistantChatOpenAI, AssistantAzureChatOpenAI, AssistantChatBedrock
        ] = LLMFactory.get_chat_model(config.llm)
        self.predictor = AutogluonTabularPredictor(config.autogluon)
        self.feature_transformers_config = get_feature_transformers_config(config)

    def describe(self) -> Dict[str, Any]:
        return {
            "predictor": self.predictor.describe(),
            "config": OmegaConf.to_container(self.config),
            "llm": self.llm.describe(),  # noqa
        }

    def handle_exception(self, stage: str, exception: Exception):
        raise Exception(str(exception), stage)

    def inference_task(self, task: TabularPredictionTask) -> TabularPredictionTask:
        logger.info("Task understanding starts...")
        task_inference_preprocessors = [
            DescriptionFileNameInference,
            DataFileNameInference,
            LabelColumnInference,
            ProblemTypeInference,
        ]

        if self.config.detect_and_drop_id_column:
            task_inference_preprocessors += [
                OutputIDColumnInference,
                TrainIDColumnInference,
                TestIDColumnInference,
            ]
        if self.config.infer_eval_metric:
            task_inference_preprocessors += [EvalMetricInference]
        for preprocessor_class in task_inference_preprocessors:
            preprocessor = preprocessor_class(llm=self.llm)
            try:
                with timeout(
                    seconds=self.config.task_preprocessors_timeout,
                    error_message=f"Task inference preprocessing timed out: {preprocessor_class}",
                ):
                    task = preprocessor.transform(task)
            except Exception as e:
                self.handle_exception(
                    f"Task inference preprocessing: {preprocessor_class}", e
                )

        bold_start = "\033[1m"
        bold_end = "\033[0m"

        logger.info(
            f"{bold_start}Total number of prompt tokens:{bold_end} {self.llm.input_}"
        )
        logger.info(
            f"{bold_start}Total number of completion tokens:{bold_end} {self.llm.output_}"
        )
        logger.info("Task understanding complete!")
        return task

    def preprocess_task(self, task: TabularPredictionTask) -> TabularPredictionTask:
        # instantiate and run task preprocessors, which infer the problem type, important filenames
        # and columns as well as the feature extractors
        task = self.inference_task(task)
        if self.feature_transformers_config:
            logger.info("Automatic feature generation starts...")
            fe_transformers = [
                instantiate(ft_config) for ft_config in self.feature_transformers_config
            ]
            for fe_transformer in fe_transformers:
                try:
                    with timeout(
                        seconds=self.config.task_preprocessors_timeout,
                        error_message=f"Task preprocessing timed out: {fe_transformer.name}",
                    ):
                        task = fe_transformer.fit_transform(task)
                except Exception as e:
                    self.handle_exception(
                        f"Task preprocessing: {fe_transformer.name}", e
                    )
            logger.info("Automatic feature generation complete!")
        else:
            logger.info("Automatic feature generation is disabled. ")
        return task

    def fit_predictor(self, task: TabularPredictionTask, time_limit: float):
        try:
            self.predictor.fit(task, time_limit=time_limit)
        except Exception as e:
            self.handle_exception("Predictor Fit", e)

    def predict(self, task: TabularPredictionTask) -> Any:
        try:
            return self.predictor.predict(task)
        except Exception as e:
            self.handle_exception("Predictor Predict", e)
