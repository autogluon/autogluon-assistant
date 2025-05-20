from .base_evaluator import BaseEvaluator
from .cls_reg_evaluator import (
    AccuracyEvaluator,
    AurocEvaluator,
    F1Evaluator,
    F1WeightedEvaluator,
    MaeEvaluator,
    MedaeEvaluator,
    NllEvaluator,
    RmseEvaluator,
    RmsleEvaluator,
    RSquaredEvaluator,
)
from .multilabel_evaluator import AverageAccuracyMultiLabelEvaluator, HammingLossMultiLabelEvaluator
from .ner_evaluator import AccuracyNEREvaluator, F1NEREvaluator
from .object_detection_evaluator import MapEvaluator
from .retrieval_evaluator import (
    MRRRetrievalEvaluator,
    NDCGRetrievalEvaluator,
    RecallRetrievalEvaluator,
)
from .sematic_segmentation_evaluator import IouEvaluator, SMeasureEvaluator
from .timeseries_evaluator import (
    MAETimeseriesEvaluator,
    MAPETimeseriesEvaluator,
    MASETimeseriesEvaluator,
    MSETimeseriesEvaluator,
    RMSETimeseriesEvaluator,
    RMSLETimeseriesEvaluator,
    RMSSETimeseriesEvaluator,
    SMAPETimeseriesEvaluator,
    SQLTimeseriesEvaluator,
    WAPETimeseriesEvaluator,
    WQLTimeseriesEvaluator,
)
from .utils import load_metadata


def get_evaluator(metric: str, problem_type: str) -> BaseEvaluator:
    classification_evaluators = {
        "accuracy": AccuracyEvaluator(),
        "f1_weighted": F1WeightedEvaluator(),
        "f1": F1Evaluator(),
        "auroc": AurocEvaluator(),
        "nll": NllEvaluator(),
    }
    regression_evaluators = {
        "rmse": RmseEvaluator(),
        "rmsle": RmsleEvaluator(),
        "mae": MaeEvaluator(),
        "medae": MedaeEvaluator(),
        "r_squared": RSquaredEvaluator(),
    }
    object_detection_evaluators = {"map": MapEvaluator()}
    semantic_segmentation_evaluators = {
        "iou": IouEvaluator(),
        "s_measure": SMeasureEvaluator(),
    }
    timeseries_evaluators = {
        "sql": SQLTimeseriesEvaluator(),
        "wql": WQLTimeseriesEvaluator(),
        "mae": MAETimeseriesEvaluator(),
        "mase": MASETimeseriesEvaluator(),
        "wape": WAPETimeseriesEvaluator(),
        "mse": MSETimeseriesEvaluator(),
        "rmse": RMSETimeseriesEvaluator(),
        "rmsle": RMSLETimeseriesEvaluator(),
        "rmsse": RMSSETimeseriesEvaluator(),
        "mape": MAPETimeseriesEvaluator(),
        "smape": SMAPETimeseriesEvaluator(),
    }
    ner_evaluators = {
        "f1": F1NEREvaluator(),
        "accuracy": AccuracyNEREvaluator(),
    }
    retrieval_evaluators = {
        "ndcg@10": NDCGRetrievalEvaluator(k=10),
        "recall@10": RecallRetrievalEvaluator(k=10),
        "mrr@10": MRRRetrievalEvaluator(k=10),
    }
    multilabel_evaluators = {
        "hamming_loss": HammingLossMultiLabelEvaluator(),
        "average_accuracy": AverageAccuracyMultiLabelEvaluator(),
    }

    metric_lower = metric.lower()
    # Select appropriate evaluator based on problem type
    if problem_type in ["binary", "multiclass"]:
        evaluators = classification_evaluators
    elif problem_type == "regression":
        evaluators = regression_evaluators
    elif problem_type == "object_detection":
        evaluators = object_detection_evaluators
    elif problem_type == "semantic_segmentation":
        evaluators = semantic_segmentation_evaluators
    elif problem_type == "timeseries":
        evaluators = timeseries_evaluators
    elif problem_type == "ner":
        evaluators = ner_evaluators
    elif problem_type == "retrieval":
        evaluators = retrieval_evaluators
    elif problem_type == "multilabel":
        evaluators = multilabel_evaluators
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")

    if metric_lower not in evaluators:
        available_metrics = ", ".join(evaluators.keys())
        raise ValueError(
            f"Unknown metric '{metric}' for problem type '{problem_type}'. "
            f"Available metrics are: {available_metrics}"
        )

    return evaluators[metric_lower]


def evaluate(pred_path: str, gt_path: str, results_path: str, metadata_path: str, agent_name: str):
    metadata = load_metadata(metadata_path)
    metric = metadata["metric_name"]
    problem_type = metadata["problem_type"]
    evaluator = get_evaluator(metric, problem_type)
    evaluator.evaluate(pred_path, gt_path, results_path, metadata_path, agent_name)
