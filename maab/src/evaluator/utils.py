import json
from typing import Dict, Any


def load_metadata(metadata_path: str) -> Dict[str, Any]:
    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        required_keys = ["dataset_name", "metric_name", "problem_type", "label_column"]
        for key in required_keys:
            if key not in metadata:
                print(f"Required key '{key}' not found in metadata")
        return metadata
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON at {metadata_path}:\n{str(e)}")
        print("Please ensure your JSON is correctly formatted.")
        raise
