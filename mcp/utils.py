"""
Utility functions for MCP server
"""

import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from constants import PROVIDER_DEFAULTS


def generate_task_output_dir() -> str:
    """
    Generate output directory path for a task.
    
    Returns:
        str: Path to output directory
    """
    # Follow the specified pattern: mlzero-{datetime}-{uuid}
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_uuid = uuid.uuid4()
    
    # Use a dedicated directory for MCP outputs
    # This ensures consistency regardless of where the server is started
    base_dir = Path.home() / ".autogluon_assistant" / "mcp_outputs"
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Create output directory
    folder_name = f"mlzero-{current_datetime}-{random_uuid}"
    output_dir = base_dir / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return str(output_dir)


def parse_credentials(credentials_text: str, provider: str) -> Optional[Dict[str, str]]:
    """
    Parse credentials from environment variable format.
    
    Args:
        credentials_text: Text containing environment variable exports
        provider: Provider name (bedrock/openai/anthropic)
        
    Returns:
        Dict of parsed credentials or None if invalid
    """
    if not credentials_text:
        return None
    
    credentials = {}
    lines = credentials_text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Remove 'export ' prefix if present
        if line.startswith('export '):
            line = line[7:]
        
        # Parse key=value
        if '=' in line:
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()
            
            # Remove quotes if present
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            elif value.startswith("'") and value.endswith("'"):
                value = value[1:-1]
            
            credentials[key] = value
    
    # Validate based on provider
    if provider == "bedrock":
        # Required fields for AWS/Bedrock
        required = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
        for field in required:
            if field not in credentials or not credentials[field]:
                return None
        
        # Set default region if not provided
        if "AWS_DEFAULT_REGION" not in credentials:
            credentials["AWS_DEFAULT_REGION"] = "us-west-2"
            
    elif provider == "openai":
        # Required fields for OpenAI
        if "OPENAI_API_KEY" not in credentials or not credentials["OPENAI_API_KEY"]:
            return None
            
    elif provider == "anthropic":
        # Required fields for Anthropic
        if "ANTHROPIC_API_KEY" not in credentials or not credentials["ANTHROPIC_API_KEY"]:
            return None
    else:
        return None
    
    return credentials


def get_default_model(provider: str) -> str:
    """
    Get default model for a provider.
    
    Args:
        provider: Provider name
        
    Returns:
        str: Default model name
    """
    return PROVIDER_DEFAULTS.get(provider, "")


def validate_path_security(path: str, allowed_prefixes: List[str]) -> bool:
    """
    Validate if a path is safe to access.
    
    Args:
        path: Path to validate
        allowed_prefixes: List of allowed path prefixes
        
    Returns:
        bool: True if path is safe
    """
    # Resolve to absolute path
    abs_path = Path(path).resolve()
    
    # Check against allowed prefixes
    for prefix in allowed_prefixes:
        prefix_path = Path(prefix).resolve()
        try:
            abs_path.relative_to(prefix_path)
            return True
        except ValueError:
            continue
    
    return False


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        str: Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def extract_iteration_from_log(log_text: str) -> Optional[int]:
    """
    Extract iteration number from log text.
    
    Args:
        log_text: Log text
        
    Returns:
        Optional[int]: Iteration number or None
    """
    match = re.search(r"Starting iteration (\d+)!", log_text)
    if match:
        return int(match.group(1))
    return None


def clean_log_markup(text: str) -> str:
    """
    Remove rich text markup from log text.
    
    Args:
        text: Log text with markup
        
    Returns:
        str: Clean text
    """
    # Remove [bold green], [bold red], etc.
    cleaned = re.sub(r'\[/?bold\s*(green|red)\]', '', text)
    return cleaned