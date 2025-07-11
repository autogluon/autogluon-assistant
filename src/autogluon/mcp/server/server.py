#!/usr/bin/env python3
"""
AutoGluon Assistant MCP Server

This server provides MCP interface for AutoGluon Assistant,
allowing remote clients to submit ML tasks and retrieve results.
"""

import logging
from pathlib import Path
from typing import Optional

from autogluon.mcp.file_handler import FileHandler
from autogluon.mcp.server.task_manager import TaskManager
from autogluon.mcp.server.utils import generate_task_output_dir
from fastmcp import FastMCP

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastMCP server instance
mcp = FastMCP("AutoGluon Assistant Server 🚀")

# Initialize handlers
file_handler = FileHandler(base_dir=Path.home() / ".autogluon_assistant" / "mcp_uploads")
task_manager = TaskManager()


# ==================== Tools ====================


@mcp.tool()
async def upload_input_folder(folder_structure: dict, file_contents: dict) -> dict:
    """
    Upload input folder from client to server.

    Args:
        folder_structure: Directory structure in JSON format
        file_contents: File contents with paths as keys and base64 content as values

    Returns:
        dict: {"success": bool, "path": str, "error": str (optional)}
    """
    try:
        server_path = file_handler.upload_folder(folder_structure, file_contents)
        logger.info(f"Input folder uploaded to: {server_path}")
        return {"success": True, "path": server_path}
    except Exception as e:
        logger.error(f"Failed to upload input folder: {str(e)}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def upload_config(filename: str, content: str) -> dict:
    """
    Upload config file to server.

    Args:
        filename: Name of the config file
        content: Base64 encoded content of the config file

    Returns:
        dict: {"success": bool, "path": str, "error": str (optional)}
    """
    try:
        server_path = file_handler.upload_single_file(filename, content)
        logger.info(f"Config file uploaded to: {server_path}")
        return {"success": True, "path": server_path}
    except Exception as e:
        logger.error(f"Failed to upload config file: {str(e)}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def start_task(
    input_dir: str,
    output_dir: str,
    config_path: Optional[str] = None,
    max_iterations: Optional[int] = 5,
    initial_user_input: Optional[str] = None,
    credentials_text: Optional[str] = None,
) -> dict:
    """
    Start AutoGluon task with given parameters.

    Args:
        input_dir: Server path to input data folder
        output_dir: Client path where results will be saved
        config_path: Server path to config file (optional)
        max_iterations: Maximum iterations (default: 5)
        initial_user_input: Initial user prompt (optional)
        credentials_text: Environment variable format credentials

    Returns:
        dict: {"success": bool, "task_id": str, "run_id": str, "error": str (optional)}
    """
    try:
        # Generate server output directory
        server_output_dir = generate_task_output_dir()

        credentials = None
        if credentials_text:
            credentials = {}
            lines = credentials_text.strip().split("\n")
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("export "):
                    line = line[7:]
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    credentials[key] = value

        # Prepare task parameters
        params = {
            "input_dir": input_dir,
            "server_output_dir": server_output_dir,
            "client_output_dir": output_dir,
            "config_path": config_path,
            "max_iterations": max_iterations,
            "initial_user_input": initial_user_input,
            "credentials": credentials,
        }

        # Start task
        result = await task_manager.start_task(params)

        if result["success"]:
            logger.info(f"Task started successfully: {result['task_id']}")
            return {
                "success": True,
                "task_id": result["task_id"],
                "run_id": result.get("run_id"),
                "position": result.get("position", 0),
            }
        else:
            return {"success": False, "error": result.get("error", "Failed to start task")}

    except Exception as e:
        logger.error(f"Failed to start task: {str(e)}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def check_status() -> dict:
    """
    Check status of current task.

    Returns:
        dict: Current task status including progress, logs, and state
    """
    try:
        status = await task_manager.check_status()
        return status
    except Exception as e:
        logger.error(f"Failed to check status: {str(e)}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def cancel_task() -> dict:
    """
    Cancel the currently running task.

    Returns:
        dict: {"success": bool, "error": str (optional)}
    """
    try:
        result = await task_manager.cancel_task()
        return result
    except Exception as e:
        logger.error(f"Failed to cancel task: {str(e)}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def list_outputs() -> dict:
    """
    List output files from completed task.

    Returns:
        dict: {"success": bool, "files": list, "error": str (optional)}
    """
    try:
        result = await task_manager.list_outputs()
        return result
    except Exception as e:
        logger.error(f"Failed to list outputs: {str(e)}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def download_file(file_path: str) -> dict:
    """
    Download a single file from server.

    Args:
        file_path: Server path to the file

    Returns:
        dict: {"success": bool, "content": str (base64), "error": str (optional)}
    """
    try:
        content = file_handler.download_file(file_path)
        return {"success": True, "content": content}
    except Exception as e:
        logger.error(f"Failed to download file: {str(e)}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def get_progress() -> dict:
    """
    Get current task progress.

    Returns:
        dict: Progress information including percentage and state
    """
    try:
        progress = await task_manager.get_progress()
        return progress
    except Exception as e:
        logger.error(f"Failed to get progress: {str(e)}")
        return {"progress": 0.0, "error": str(e)}


@mcp.tool()
async def cleanup_output(output_dir: str) -> dict:
    """
    Clean up output directory on server after successful download.

    Args:
        output_dir: Server path to output directory

    Returns:
        dict: {"success": bool, "error": str (optional)}
    """
    try:
        import shutil

        path = Path(output_dir)

        # Security check
        if not file_handler._is_safe_path(path):
            return {"success": False, "error": f"Access denied: {output_dir}"}

        if not path.exists():
            return {"success": False, "error": f"Directory not found: {output_dir}"}

        # Remove the directory
        shutil.rmtree(path)
        logger.info(f"Cleaned up output directory: {output_dir}")

        return {"success": True}
    except Exception as e:
        logger.error(f"Failed to cleanup output: {str(e)}")
        return {"success": False, "error": str(e)}


# ==================== Prompts ====================


@mcp.prompt()
def autogluon_workflow() -> str:
    """Complete AutoGluon workflow guide"""
    return """
    AutoGluon Assistant Workflow:
    
    1. Upload Phase:
       - First upload your input folder using upload_input_folder
       - The folder_structure should be a JSON representation of your directory tree
       - The file_contents should be a dict with relative paths as keys and base64 content as values
       - Optionally upload a config file using upload_config
       
    2. Start Task:
       - Call start_task with the server paths returned from upload
       - Specify your desired output directory on the client side
       - Choose provider (bedrock/openai/anthropic) and optionally specify model
       - Provide credentials in environment variable format if needed
       - Set max_iterations and whether you want interactive input
       
    3. Monitor Progress:
       - Use check_status to monitor task progress
       - The response includes current state, logs, and progress
       - If waiting_for_input is true, use send_input to provide response
       
    4. Download Results:
       - Once task is completed, use list_outputs to see available files
       - Use download_file to retrieve each file you need
       - Files are returned as base64 encoded content
       
    Notes:
    - Only one task can run at a time
    - Tasks run with INFO log level by default
    - Archive extraction happens automatically if needed
    """


@mcp.prompt()
def credentials_format() -> str:
    """Credentials format guide"""
    return """
    Credentials Format (environment variable style):
    
    For AWS/Bedrock:
    ```
    export AWS_ACCESS_KEY_ID="AKIAIOSFODNN7EXAMPLE"
    export AWS_SECRET_ACCESS_KEY="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    export AWS_SESSION_TOKEN="..."  # if using temporary credentials
    export AWS_DEFAULT_REGION="us-west-2"
    ```
    
    For OpenAI:
    ```
    export OPENAI_API_KEY="sk-..."
    ```
    
    For Anthropic:
    ```
    export ANTHROPIC_API_KEY="sk-ant-..."
    ```
    
    Pass the entire text block (including 'export' statements) as credentials_text parameter.
    """


@mcp.prompt()
def folder_structure_format() -> str:
    """Folder structure format guide"""
    return """
    Folder Structure Format:
    
    The folder_structure parameter should be a JSON object like:
    ```json
    {
        "type": "directory",
        "name": "root",
        "children": [
            {
                "type": "file",
                "name": "train.csv",
                "size": 1024,
                "path": "train.csv"
            },
            {
                "type": "directory",
                "name": "data",
                "children": [
                    {
                        "type": "file",
                        "name": "test.csv",
                        "size": 2048,
                        "path": "data/test.csv"
                    }
                ]
            }
        ]
    }
    ```
    
    The file_contents should be:
    ```json
    {
        "train.csv": "base64_encoded_content_here",
        "data/test.csv": "base64_encoded_content_here"
    }
    ```
    """


# ==================== Resources ====================


@mcp.resource("task://current")
async def get_current_task() -> dict:
    """Get current running task information"""
    current = await task_manager.get_current_task_info()
    if current:
        return {
            "task_id": current.get("task_id"),
            "run_id": current.get("run_id"),
            "state": current.get("state"),
            "started_at": current.get("started_at"),
            "input_dir": current.get("input_dir"),
            "output_dir": current.get("server_output_dir"),
        }
    else:
        return {"message": "No task currently running"}


# ==================== Main ====================

if __name__ == "__main__":
    # Run with streamable HTTP transport by default
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)
