from copy import deepcopy

# Task Inference
NO_FILE_IDENTIFIED = "NO_FILE_IDENTIFIED"
NO_ID_COLUMN_IDENTIFIED = "NO_ID_COLUMN_IDENTIFIED"

# Supported File Types
TEXT_EXTENSIONS = [".txt", ".md", ".json", ".yml", ".yaml", ".xml", ".log"]

# Data types
TRAIN = "train"
TEST = "test"
OUTPUT = "output"

# Problem types
BINARY = "binary"
MULTICLASS = "multiclass"
REGRESSION = "regression"
PROBLEM_TYPES = [BINARY, MULTICLASS, REGRESSION]
CLASSIFICATION_PROBLEM_TYPES = [BINARY, MULTICLASS]

# Presets/Configs
CONFIGS = "configs"
MEDIUM_QUALITY = "medium_quality"
HIGH_QUALITY = "high_quality"
BEST_QUALITY = "best_quality"
DEFAULT_QUALITY = BEST_QUALITY
PRESETS = [MEDIUM_QUALITY, HIGH_QUALITY, BEST_QUALITY]

# Metrics
ROC_AUC = "roc_auc"
LOG_LOSS = "log_loss"
ACCURACY = "accuracy"
F1 = "f1"
QUADRATIC_KAPPA = "quadratic_kappa"
BALANCED_ACCURACY = "balanced_accuracy"
ROOT_MEAN_SQUARED_ERROR = "root_mean_squared_error"
MEAN_SQUARED_ERROR = "mean_squared_error"
MEAN_ABSOLUTE_ERROR = "mean_absolute_error"
R2 = "r2"
ROOT_MEAN_SQUARED_LOGARITHMIC_ERROR = "root_mean_squared_logarithmic_error"

CLASSIFICATION_PROBA_EVAL_METRIC = [ROC_AUC, LOG_LOSS, F1]

METRICS_DESCRIPTION = {
    ROC_AUC: "Area under the receiver operating characteristics (ROC) curve",
    LOG_LOSS: "Log loss, also known as logarithmic loss",
    ACCURACY: "Accuracy",
    F1: "F1 score",
    QUADRATIC_KAPPA: "Quadratic kappa, i.e., the Cohen kappa metric",
    BALANCED_ACCURACY: "Balanced accuracy, i.e., the arithmetic mean of sensitivity and specificity",
    ROOT_MEAN_SQUARED_ERROR: "Root mean squared error (RMSE)",
    MEAN_SQUARED_ERROR: "Mean squared error (MSE)",
    MEAN_ABSOLUTE_ERROR: "Mean absolute error (MAE)",
    R2: "R-squared",
    ROOT_MEAN_SQUARED_LOGARITHMIC_ERROR: "Root mean squared logarithmic error (RMSLE)",
}

METRICS_BY_PROBLEM_TYPE = {
    BINARY: [ROC_AUC, LOG_LOSS, ACCURACY, F1, QUADRATIC_KAPPA, BALANCED_ACCURACY],
    MULTICLASS: [ROC_AUC, LOG_LOSS, ACCURACY, F1, QUADRATIC_KAPPA, BALANCED_ACCURACY],
    REGRESSION: [
        ROOT_MEAN_SQUARED_ERROR,
        MEAN_SQUARED_ERROR,
        MEAN_ABSOLUTE_ERROR,
        R2,
        ROOT_MEAN_SQUARED_LOGARITHMIC_ERROR,
    ],
}

PREFERED_METRIC_BY_PROBLEM_TYPE = {
    BINARY: ROC_AUC,
    MULTICLASS: ROC_AUC,
    REGRESSION: ROOT_MEAN_SQUARED_ERROR,
}

WHITE_LIST_LLM = [
    "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "meta.llama3-1-405b-instruct-v1:0",
    "anthropic.claude-3-5-haiku-20241022-v1:0",
    "gpt-4o-2024-08-06",
    "anthropic.claude-3-5-sonnet-20240620-v1:0",
]

#  The below constants are for Autogluon-Assistant UI
BASE_DATA_DIR = "./user_data"


# Preset configurations
PRESET_DEFAULT_CONFIG = {
    "Best Quality": {"time_limit": "4 hrs", "feature_generation": False},
    "High Quality": {"time_limit": "1 hr", "feature_generation": False},
    "Medium Quality": {"time_limit": "10 mins", "feature_generation": False},
}
DEFAULT_PRESET = "Medium Quality"

PRESET_MAPPING = {
    "Best Quality": "best_quality",
    "High Quality": "high_quality",
    "Medium Quality": "medium_quality",
}
PRESET_OPTIONS = ["Best Quality", "High Quality", "Medium Quality"]

# Time limit configurations (in seconds)
TIME_LIMIT_MAPPING = {
    "3 mins": 180,
    "10 mins": 600,
    "30 mins": 1800,
    "1 hr": 3600,
    "2 hrs": 7200,
    "4 hrs": 14400,
}

DEFAULT_TIME_LIMIT = "10 mins"

TIME_LIMIT_OPTIONS = ["3 mins", "10 mins", "30 mins", "1 hr", "2 hrs", "4 hrs"]

# LLM configurations
LLM_MAPPING = {
    "Claude 3.5 with Amazon Bedrock": "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "GPT 4o": "gpt-4o-2024-08-06",
    "GPT 4o (Azure)": "gpt-4o-2024-08-06",
}

LLM_OPTIONS = ["Claude 3.5 with Amazon Bedrock", "GPT 4o", "GPT 4o (Azure)"]

# Provider configuration
PROVIDER_MAPPING = {
    "Claude 3.5 with Amazon Bedrock": "bedrock",
    "GPT 4o": "openai",
    "GPT 4o (Azure)": "azure",
}

INITIAL_STAGE = {
    "Task Understanding": [],
    "Feature Generation": [],
    "Model Training": [],
    "Prediction": [],
}
# Initial Session state
DEFAULT_SESSION_VALUES = {
    "config_overrides": [],
    "preset": DEFAULT_PRESET,
    "time_limit": DEFAULT_TIME_LIMIT,
    "llm": None,
    "pid": None,
    "logs": "",
    "process": None,
    "clicked": False,
    "task_running": False,
    "output_file": None,
    "output_filename": None,
    "task_description": "",
    "sample_description": "",
    "return_code": None,
    "task_canceled": False,
    "uploaded_files": {},
    "sample_files": {},
    "selected_dataset": None,
    "sample_dataset_dir": None,
    "description_uploader_key": 0,
    "sample_dataset_selector": None,
    "current_stage": None,
    "feature_generation": False,
    "stage_status": {},
    "show_remaining_time": False,
    "model_path": None,
    "elapsed_time": 0,
    "progress_bar": None,
    "increment": 2,
    "zip_path": None,
    "stage_container": deepcopy(INITIAL_STAGE),
    "start_time": None,
    "remaining_time": 0,
    "start_model_train_time": 0,
}

# Message to display different logging stage
STATUS_BAR_STAGE = {
    "Task loaded!": 10,
    "Model training starts": 25,
    "Fitting model": 50,
    "AutoGluon training complete": 80,
    "Prediction starts": 90,
}

STAGE_COMPLETE_SIGNAL = [
    "Task understanding complete",
    "Automatic feature generation complete",
    "Model training complete",
    "Prediction complete",
]

# Stage Names
STAGE_TASK_UNDERSTANDING = "Task Understanding"
STAGE_FEATURE_GENERATION = "Feature Generation"
STAGE_MODEL_TRAINING = "Model Training"
STAGE_PREDICTION = "Prediction"

# Log Messages
MSG_TASK_UNDERSTANDING = "Task understanding starts"
MSG_FEATURE_GENERATION = "Automatic feature generation starts"
MSG_MODEL_TRAINING = "Model training starts"
MSG_PREDICTION = "Prediction starts"

# Mapping
STAGE_MESSAGES = {
    MSG_TASK_UNDERSTANDING: STAGE_TASK_UNDERSTANDING,
    MSG_FEATURE_GENERATION: STAGE_FEATURE_GENERATION,
    MSG_MODEL_TRAINING: STAGE_MODEL_TRAINING,
    MSG_PREDICTION: STAGE_PREDICTION,
}
# DataSet Options
DATASET_OPTIONS = ["Sample Dataset", "Upload Dataset"]

# Captions under DataSet Options
CAPTIONS = ["Run with sample dataset", "Upload Train (Required), Test (Required) and Output (Optional) Dataset"]

DEMO_URL = "https://automl-mm-bench.s3.amazonaws.com/autogluon-assistant/aga-kaggle-demo.mp4"

LOGO_PATH = "static/page_icon.png"
SUCCESS_MESSAGE = """
        🎉🎉Task completed successfully! If you found this useful, please consider:
        ⭐ [Starring our repository](https://github.com/autogluon/autogluon-assistant)
        """
S3_URL = "https://automl-mm-bench.s3.us-east-1.amazonaws.com/autogluon-assistant/sample_dataset.zip"
LOCAL_ZIP_PATH = "sample_data.zip"
EXTRACT_DIR = "sample_dataset"
IGNORED_MESSAGES = [
    "Failed to identify the sample_submission_data of the task, it is set to None.",
    "Too many requests, please wait before trying again",
]
