#!/usr/bin/env python3
"""
MCP Server that exposes the complete AutoGluon pipeline as a single tool
"""

import asyncio
import base64
import json
from pathlib import Path
from typing import Optional

from autogluon.mcp.file_handler import analyze_folder, read_files_for_upload
from fastmcp import Client, FastMCP

# Create MCP server
mcp = FastMCP("AutoGluon Pipeline Server")


def parse_mcp_response(response):
    """Parse FastMCP response format"""
    if isinstance(response, list) and len(response) > 0:
        text_content = response[0].text
        return json.loads(text_content)
    return response


def count_files(structure: dict) -> int:
    """Count files in folder structure"""
    if structure["type"] == "file":
        return 1
    else:
        return sum(count_files(child) for child in structure.get("children", []))


def load_credentials_from_file(file_path: str) -> str:
    """Load credentials from file"""
    path = Path(file_path)
    if path.exists():
        return path.read_text()
    return ""


@mcp.tool()
async def run_autogluon_pipeline(
    input_folder: str,
    output_folder: str,
    server_url: str = "http://127.0.0.1:8000/mcp/",
    verbosity: str = "info",
    config_file: Optional[str] = None,
    max_iterations: int = 5,
    init_prompt: Optional[str] = None,
    creds_path: Optional[str] = None,
    cleanup_server: bool = True,
) -> dict:
    """
    Run complete AutoGluon pipeline from data upload to results download.

    Use this tool when:
    - User wants to train a machine learning model using AutoGluon
    - User has data files (CSV, Parquet, etc.) and wants automated ML
    - User mentions AutoML, AutoGluon, or automatic model training
    - User asks to analyze/predict/classify data using ML

    This tool will upload data, run AutoGluon training, and download results automatically.

    Args:
        input_folder: Local path to input data
        output_folder: Local path where results will be saved
        server_url: MCP server URL (e.g., https://your-server.ngrok.app)
        verbosity: Log level - "brief", "info", or "detail"
        config_file: Optional path to config file
        max_iterations: Maximum iterations (default: 5)
        init_prompt: Initial user prompt (optional)
        creds_path: Path to credentials file (optional)
        cleanup_server: Whether to clean up server files after download

    Returns:
        dict: Execution results with brief logs and output file paths
    """
    # Initialize log collectors
    all_logs = []
    brief_logs = []

    def log(message: str, level: str = "INFO"):
        """Log message and collect based on verbosity"""
        all_logs.append({"level": level, "text": message})

        # Collect brief logs for return
        if level in ["BRIEF", "ERROR"]:
            brief_logs.append({"level": level, "text": message})

        # Print based on verbosity setting
        if verbosity == "brief" and level in ["BRIEF", "ERROR"]:
            print(f"[{level}] {message}")
        elif verbosity == "info" and level in ["BRIEF", "INFO", "ERROR"]:
            print(f"[{level}] {message}")
        elif verbosity == "detail":
            print(f"[{level}] {message}")

    # Load credentials if provided
    credentials_text = None
    if creds_path:
        credentials_text = load_credentials_from_file(creds_path)
        if not credentials_text:
            log(f"Warning: Could not load credentials from {creds_path}", "ERROR")

    # Create client
    if not server_url.endswith("/mcp"):
        server_url = server_url.rstrip("/") + "/mcp"
    client = Client(server_url)

    try:
        async with client:
            log("Connected to AutoGluon MCP Server", "BRIEF")

            # 1. Upload input folder
            log(f"Uploading input folder: {input_folder}", "BRIEF")

            # Analyze folder structure
            folder_structure = analyze_folder(input_folder)
            log(f"Found {count_files(folder_structure)} files", "INFO")

            # Read file contents
            file_contents = read_files_for_upload(input_folder)
            total_size = sum(len(c) for c in file_contents.values()) / 1024 / 1024
            log(f"Total size: {total_size:.1f} MB", "INFO")

            # Upload folder
            result = await client.call_tool(
                "upload_input_folder", {"folder_structure": folder_structure, "file_contents": file_contents}
            )
            result = parse_mcp_response(result)

            if not result.get("success", False):
                error_msg = result.get("error", "Upload failed")
                log(f"ERROR: {error_msg}", "ERROR")
                return {"success": False, "error": error_msg, "logs": brief_logs}

            server_input_dir = result["path"]
            log(f"Uploaded to: {server_input_dir}", "INFO")

            # 2. Upload config file if provided
            server_config_path = None
            if config_file:
                log(f"Uploading config file: {config_file}", "INFO")

                config_path = Path(config_file)
                if not config_path.exists():
                    log("ERROR: Config file not found", "ERROR")
                    return {"success": False, "error": "Config file not found", "logs": brief_logs}

                # Read and encode config
                config_content = config_path.read_bytes()
                config_b64 = base64.b64encode(config_content).decode("utf-8")

                result = await client.call_tool("upload_config", {"filename": config_path.name, "content": config_b64})
                result = parse_mcp_response(result)

                if not result["success"]:
                    error_msg = result.get("error", "Upload failed")
                    log(f"ERROR: {error_msg}", "ERROR")
                    return {"success": False, "error": error_msg, "logs": brief_logs}

                server_config_path = result["path"]
                log(f"Config uploaded to: {server_config_path}", "INFO")

            # 3. Start task
            log("Starting AutoGluon task", "BRIEF")
            log(f"Max iterations: {max_iterations}", "INFO")
            if init_prompt:
                log(f"Initial prompt: {init_prompt}", "INFO")

            result = await client.call_tool(
                "start_task",
                {
                    "input_dir": server_input_dir,
                    "output_dir": output_folder,
                    "config_path": server_config_path,
                    "max_iterations": max_iterations,
                    "initial_user_input": init_prompt,
                    "credentials_text": credentials_text,
                },
            )
            result = parse_mcp_response(result)

            if not result["success"]:
                error_msg = result.get("error", "Failed to start task")
                log(f"ERROR: {error_msg}", "ERROR")
                return {"success": False, "error": error_msg, "logs": brief_logs}

            task_id = result["task_id"]
            position = result.get("position", 0)

            log(f"Task started: {task_id}", "BRIEF")
            if position > 0:
                log(f"Queue position: {position}", "INFO")

            # 4. Monitor progress
            log("Monitoring task progress...", "INFO")

            last_log_count = 0
            while True:
                # Check status
                status = await client.call_tool("check_status", {})
                status = parse_mcp_response(status)

                if not status["success"]:
                    error_msg = status.get("error", "Status check failed")
                    log(f"ERROR: {error_msg}", "ERROR")
                    break

                # Process new logs
                logs = status.get("logs", [])
                new_logs = logs[last_log_count:]
                for task_log in new_logs:
                    if isinstance(task_log, dict):
                        level = task_log.get("level", "INFO")
                        text = task_log.get("text", "")
                        # Map task log levels to our log function
                        if level == "BRIEF":
                            log(text, "BRIEF")
                        elif level == "ERROR":
                            log(text, "ERROR")
                        else:
                            log(text, "DETAIL")
                    else:
                        log(str(task_log), "DETAIL")
                last_log_count = len(logs)

                # Check if completed
                if status.get("state") == "completed":
                    log("Task completed successfully!", "BRIEF")
                    break
                elif status.get("state") == "failed":
                    log("Task failed!", "ERROR")
                    break

                # Update progress
                progress_info = await client.call_tool("get_progress", {})
                progress_info = parse_mcp_response(progress_info)
                if isinstance(progress_info, dict):
                    progress = progress_info.get("progress", 0.0)
                    log(f"Progress: {progress * 100:.1f}%", "DETAIL")

                # Wait before next check
                await asyncio.sleep(2)

            # 5. Download results
            log(f"Downloading results to: {output_folder}", "BRIEF")

            # List output files
            result = await client.call_tool("list_outputs", {})
            result = parse_mcp_response(result)

            if not result["success"]:
                error_msg = result.get("error", "Failed to list outputs")
                log(f"ERROR: {error_msg}", "ERROR")
                return {"success": False, "error": error_msg, "logs": brief_logs}

            files = result["files"]
            output_dir = result.get("output_dir")
            log(f"Found {len(files)} output files", "INFO")

            # Create output directory
            output_path = Path(output_folder)
            output_path.mkdir(parents=True, exist_ok=True)

            # Extract folder name from server path
            if output_dir:
                server_folder_name = Path(output_dir).name
                local_output_base = output_path / server_folder_name
                local_output_base.mkdir(exist_ok=True)
            else:
                local_output_base = output_path

            # Download each file
            downloaded_files = []
            for file_path in files:
                log(f"Downloading: {file_path}", "DETAIL")

                result = await client.call_tool("download_file", {"file_path": file_path})
                result = parse_mcp_response(result)

                if not result["success"]:
                    log(f"ERROR: Failed to download {file_path}", "ERROR")
                    continue

                # Decode and save file
                content = base64.b64decode(result["content"])

                # Preserve directory structure
                if output_dir and file_path.startswith(output_dir):
                    rel_path = Path(file_path).relative_to(output_dir)
                else:
                    rel_path = Path(file_path).name

                local_path = local_output_base / rel_path
                local_path.parent.mkdir(parents=True, exist_ok=True)
                local_path.write_bytes(content)

                downloaded_files.append(str(local_path))
                log(f"Saved to: {local_path}", "INFO")

            log(f"All files downloaded to: {local_output_base}", "BRIEF")

            # Optionally clean up server files
            if cleanup_server and output_dir:
                log("Cleaning up server files...", "INFO")
                result = await client.call_tool("cleanup_output", {"output_dir": output_dir})
                result = parse_mcp_response(result)

                if result.get("success"):
                    log("Server files cleaned up", "INFO")
                else:
                    log(f"Cleanup failed: {result.get('error', 'Unknown error')}", "ERROR")

            # Return results
            return {
                "success": True,
                "logs": brief_logs,
                "output_directory": str(local_output_base),
                "output_files": downloaded_files,
                "task_id": task_id,
            }

    except Exception as e:
        error_msg = str(e)
        log(f"ERROR: {error_msg}", "ERROR")
        return {"success": False, "error": error_msg, "logs": brief_logs}


def main():
    """Entry point for mlzero-mcp-client command"""
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8005, path="/mcp")


if __name__ == "__main__":
    main()
