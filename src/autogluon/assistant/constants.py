VALID_CODING_LANGUAGES = ["python", "bash"]
LOGO_PATH = "static/page_icon.png"
DEMO_URL = "https://automl-mm-bench.s3.amazonaws.com/autogluon-assistant/aga-kaggle-demo.mp4"
MODEL_INFO_LEVEL = 19
BRIEF_LEVEL = 25

API_URL = "http://localhost:5000/api"

DEFAULT_SESSION_VALUES = {
    "config_overrides": [],
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
    "start_time": None,
    "remaining_time": 0,
    "start_model_train_time": 0,
}

INITIAL_STAGE = {
    "Task Understanding": [],
    "Feature Generation": [],
    "Model Training": [],
    "Prediction": [],
}

SUCCESS_MESSAGE = """
        🎉🎉Task completed successfully! If you found this useful, please consider:
        ⭐ [Starring our repository](https://github.com/autogluon/autogluon-assistant)
        """

STATUS_BAR_STAGE = {
    "Task loaded!": 10,
    "Model training starts": 25,
    "Fitting model": 50,
    "AutoGluon training complete": 80,
    "Prediction starts": 90,
}

IGNORED_MESSAGES = [
    "Failed to identify the sample_submission_data of the task, it is set to None.",
    "Too many requests, please wait before trying again",
]

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

STAGE_MESSAGES = {
    MSG_TASK_UNDERSTANDING: STAGE_TASK_UNDERSTANDING,
    MSG_FEATURE_GENERATION: STAGE_FEATURE_GENERATION,
    MSG_MODEL_TRAINING: STAGE_MODEL_TRAINING,
    MSG_PREDICTION: STAGE_PREDICTION,
}

TIME_LIMIT_MAPPING = {
    "3 mins": 180,
    "10 mins": 600,
    "30 mins": 1800,
    "1 hr": 3600,
    "2 hrs": 7200,
    "4 hrs": 14400,
}

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

BRIEF_STAGE_MESSAGES = {
    "Reading file":       "Data Loading",
    "Splitting data":     "Data Splitting",
    "Cleaning data":      "Data Cleaning",
    "Feature generation": "Feature Engineering",
    "Training model":     "Model Training",
    "Evaluating model":   "Model Evaluation",
    # ... add other brief-message keys as needed
}