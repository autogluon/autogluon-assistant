VALID_CODING_LANGUAGES = ["python", "bash"]
LOGO_PATH = "static/page_icon.png"
DEMO_URL = "https://automl-mm-bench.s3.amazonaws.com/autogluon-assistant/aga-kaggle-demo.mp4"
MODEL_INFO_LEVEL = 19
BRIEF_LEVEL = 25

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
