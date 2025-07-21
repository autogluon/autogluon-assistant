#!/usr/bin/env python3
"""
AutoGluon Assistant MCP Server

This server provides MCP interface for AutoGluon Assistant,
allowing remote clients to submit ML tasks and retrieve results.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

from autogluon.mcp.server.task_manager import TaskManager
from autogluon.mcp.server.utils import generate_task_output_dir
from fastmcp import FastMCP

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastMCP server instance
mcp = FastMCP("AutoGluon Assistant Server 🚀")

# Initialize handlers
task_manager = TaskManager()


# ==================== Tools ====================


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
        if not path.exists():
            return {"success": False, "error": f"Directory not found: {output_dir}"}

        # Remove the directory
        shutil.rmtree(path)
        logger.info(f"Cleaned up output directory: {output_dir}")

        return {"success": True}
    except Exception as e:
        logger.error(f"Failed to cleanup output: {str(e)}")
        return {"success": False, "error": str(e)}


# ==================== Main ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoGluon Assistant MCP Server")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host (default: 0.0.0.0)")
    args = parser.parse_args()
    
    # Run with streamable HTTP transport
    mcp.run(transport="streamable-http", host=args.host, port=args.port)
