#!/usr/bin/env python3
"""
MCP Server that exposes the complete AutoGluon pipeline as a single tool
"""

import argparse
import asyncio
import json
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastmcp import Client, FastMCP

# Create MCP server
mcp = FastMCP("AutoGluon Assistant MCP Server")


def parse_mcp_response(response):
    """Parse FastMCP response format"""
    if isinstance(response, list) and len(response) > 0:
        text_content = response[0].text
        return json.loads(text_content)
    return response


def load_credentials_from_file(file_path: str) -> str:
    """Load credentials from file"""
    path = Path(file_path)
    if path.exists():
        return path.read_text()
    return ""


@mcp.tool()
async def run_autogluon_assistant(
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
    This tool transforms raw multimodal data into high-quality ML solutions


    Use this tool when:
        user provides a folder containing a dataset and needs to obtain ML solutions.


    This tool will upload data, run AutoGluon training, and download results automatically.
    When you decide to use this tool but the user only provides an input folder without specifying an output folder, prompt the user to provide an output folder.


    Args:
        input_folder: Local path to input data (required)
        output_folder: Local path where results will be saved (required)
        server_url: MCP server URL (e.g., https://your-server.ngrok.app) (default: http://127.0.0.1:8000/mcp/)
        verbosity: Log level - "brief", "info", or "detail" (default: info)
        config_file: Optional path to config file (optional)
        max_iterations: Maximum iterations (default: 5)
        init_prompt: Initial user prompt (optional)
        creds_path: Path to credentials file (optional)
        cleanup_server: Whether to clean up server files after download (default: True)


    Returns:
        dict: Execution results with brief logs and output file paths
    """

    server_info_path = Path(__file__).parent.parent / "server_info.txt"
    RSYNC_SERVER = ""
    if server_info_path.exists():
        RSYNC_SERVER = server_info_path.read_text().strip()

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
            log("Connected to AutoGluon Assistant MCP Server", "BRIEF")

            # 1. Upload input folder using rsync
            log(f"Uploading input folder: {input_folder}", "BRIEF")

            # Generate directory path based on local/remote mode
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = uuid.uuid4().hex[:8]
            upload_dirname = f"upload_{timestamp}_{unique_id}"

            if RSYNC_SERVER:  # Remote transfer
                # Use ~ for remote path - let remote system resolve it
                remote_user = RSYNC_SERVER.split("@")[0] if "@" in RSYNC_SERVER else "ubuntu"
                remote_base = f"~/.autogluon_assistant/mcp_uploads/{upload_dirname}"
                server_input_dir = f"/home/{remote_user}/.autogluon_assistant/mcp_uploads/{upload_dirname}"
                rsync_dest = f"{RSYNC_SERVER}:{remote_base}/"
                log(f"Remote transfer to: {RSYNC_SERVER}", "INFO")
            else:  # Local transfer
                # Use Path.home() for cross-platform compatibility
                local_base = Path.home() / ".autogluon_assistant" / "mcp_uploads" / upload_dirname
                local_base.mkdir(parents=True, exist_ok=True)
                server_input_dir = str(local_base)
                rsync_dest = f"{local_base}/"
                log("Local transfer mode", "INFO")

            # Rsync upload
            rsync_cmd = [
                "rsync",
                "-avz",
                "--progress",
                f"{input_folder.rstrip('/')}/",  # Ensure trailing slash for content only
                rsync_dest,
            ]

            log(f"Running: {' '.join(rsync_cmd)}", "INFO")
            rsync_result = subprocess.run(rsync_cmd, capture_output=True, text=True)

            if rsync_result.returncode != 0:
                log(f"ERROR: rsync failed: {rsync_result.stderr}", "ERROR")
                return {"success": False, "error": f"rsync failed: {rsync_result.stderr}", "logs": brief_logs}

            log(f"Uploaded to: {server_input_dir}", "INFO")

            # 2. Upload config file if provided using rsync
            server_config_path = None
            if config_file:
                log(f"Uploading config file: {config_file}", "INFO")

                config_path = Path(config_file)
                if not config_path.exists():
                    log("ERROR: Config file not found", "ERROR")
                    return {"success": False, "error": "Config file not found", "logs": brief_logs}

                # Generate config directory path
                config_dirname = f"config_{timestamp}_{unique_id}"

                if RSYNC_SERVER:  # Remote transfer
                    remote_user = RSYNC_SERVER.split("@")[0] if "@" in RSYNC_SERVER else "ubuntu"
                    remote_config_dir = f"~/.autogluon_assistant/mcp_uploads/{config_dirname}"
                    server_config_path = (
                        f"/home/{remote_user}/.autogluon_assistant/mcp_uploads/{config_dirname}/{config_path.name}"
                    )
                    rsync_config_dest = f"{RSYNC_SERVER}:{remote_config_dir}/"
                else:  # Local transfer
                    local_config_dir = Path.home() / ".autogluon_assistant" / "mcp_uploads" / config_dirname
                    local_config_dir.mkdir(parents=True, exist_ok=True)
                    server_config_path = str(local_config_dir / config_path.name)
                    rsync_config_dest = f"{local_config_dir}/"

                # Rsync upload config file
                rsync_cmd = ["rsync", "-avz", "--progress", str(config_file), rsync_config_dest]

                log(f"Running: {' '.join(rsync_cmd)}", "INFO")
                rsync_result = subprocess.run(rsync_cmd, capture_output=True, text=True)

                if rsync_result.returncode != 0:
                    log(f"ERROR: rsync config failed: {rsync_result.stderr}", "ERROR")
                    return {
                        "success": False,
                        "error": f"rsync config failed: {rsync_result.stderr}",
                        "logs": brief_logs,
                    }

                log(f"Config uploaded to: {server_config_path}", "INFO")

            # 3. Start task
            log("Starting AutoGluon task", "BRIEF")
            log(f"Max iterations: {max_iterations}", "INFO")
            if init_prompt:
                log(f"Initial prompt: {init_prompt}", "INFO")

            task_result = await client.call_tool(
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
            task_result = parse_mcp_response(task_result)

            if not task_result["success"]:
                error_msg = task_result.get("error", "Failed to start task")
                log(f"ERROR: {error_msg}", "ERROR")
                return {"success": False, "error": error_msg, "logs": brief_logs}

            task_id = task_result["task_id"]
            position = task_result.get("position", 0)

            log(f"Task started: {task_id}", "BRIEF")
            if position > 0:
                log(f"Queue position: {position}", "INFO")

            # 4. Monitor progress
            log("Monitoring task progress...", "INFO")

            last_log_count = 0
            output_dir = None  # Will be set when task completes

            while True:
                # Check status
                status_result = await client.call_tool("check_status", {})
                status_result = parse_mcp_response(status_result)

                if not status_result["success"]:
                    error_msg = status_result.get("error", "Status check failed")
                    log(f"ERROR: {error_msg}", "ERROR")
                    break

                # Process new logs
                logs = status_result.get("logs", [])
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
                if status_result.get("state") == "completed":
                    output_dir = status_result.get("output_dir")
                    log("Task completed successfully!", "BRIEF")
                    break
                elif status_result.get("state") == "failed":
                    log("Task failed!", "ERROR")
                    break

                # Update progress
                progress_result = await client.call_tool("get_progress", {})
                progress_result = parse_mcp_response(progress_result)
                if isinstance(progress_result, dict):
                    progress = progress_result.get("progress", 0.0)
                    log(f"Progress: {progress * 100:.1f}%", "DETAIL")

                # Wait before next check
                await asyncio.sleep(2)

            # 5. Download results
            log(f"Downloading results to: {output_folder}", "BRIEF")

            # Get server output directory if not already from status
            if not output_dir:
                outputs_result = await client.call_tool("list_outputs", {})
                outputs_result = parse_mcp_response(outputs_result)

                if not outputs_result["success"]:
                    error_msg = outputs_result.get("error", "Failed to get outputs")
                    log(f"ERROR: {error_msg}", "ERROR")
                    return {"success": False, "error": error_msg, "logs": brief_logs}

                output_dir = outputs_result.get("output_dir")

            if not output_dir:
                log("ERROR: No output directory found", "ERROR")
                return {"success": False, "error": "No output directory", "logs": brief_logs}

            log(f"Server output directory: {output_dir}", "INFO")

            # Create local output directory
            output_path = Path(output_folder)
            output_path.mkdir(parents=True, exist_ok=True)

            # Rsync download from server to client
            if RSYNC_SERVER:  # Remote transfer
                rsync_source = f"{RSYNC_SERVER}:{output_dir}"
            else:  # Local transfer
                rsync_source = output_dir

            rsync_cmd = [
                "rsync",
                "-avz",
                "--progress",
                rsync_source,  # No trailing slash - copy the folder itself
                f"{output_folder}/",
            ]

            log(f"Running: {' '.join(rsync_cmd)}", "INFO")
            rsync_result = subprocess.run(rsync_cmd, capture_output=True, text=True)

            if rsync_result.returncode != 0:
                log(f"ERROR: rsync failed: {rsync_result.stderr}", "ERROR")
                return {"success": False, "error": f"rsync download failed: {rsync_result.stderr}", "logs": brief_logs}

            # List downloaded files
            server_folder_name = Path(output_dir).name
            local_output_base = output_path / server_folder_name
            downloaded_files = []
            if local_output_base.exists():
                for file_path in local_output_base.rglob("*"):
                    if file_path.is_file():
                        downloaded_files.append(str(file_path))

            log(f"All files downloaded to: {local_output_base}", "BRIEF")

            # Optionally clean up server files
            if cleanup_server and output_dir:
                log("Cleaning up server files...", "INFO")
                cleanup_result = await client.call_tool("cleanup_output", {"output_dir": output_dir})
                cleanup_result = parse_mcp_response(cleanup_result)

                if cleanup_result.get("success"):
                    log("Server files cleaned up", "INFO")
                else:
                    log(f"Cleanup failed: {cleanup_result.get('error', 'Unknown error')}", "ERROR")

            # Return results
            return {
                "success": True,
                "logs": brief_logs,
                "output_directory": str(local_output_base),
                "task_id": task_id,
            }

    except Exception as e:
        error_msg = str(e)
        log(f"ERROR: {error_msg}", "ERROR")
        return {"success": False, "error": error_msg, "logs": brief_logs}


def main():
    """Entry point for mlzero-mcp-client command"""
    parser = argparse.ArgumentParser(description="AutoGluon MCP Client")
    parser.add_argument(
        "--server", "-s", type=str, default="", help="Rsync server (e.g., ubuntu@ec2-ip). Leave empty for local mode."
    )
    parser.add_argument("--port", "-p", type=int, default=8005, help="MCP server port (default: 8005)")
    args = parser.parse_args()

    server_info_path = Path(__file__).parent.parent / "server_info.txt"
    server_info_path.write_text(args.server)

    mcp.run(transport="streamable-http", host="0.0.0.0", port=args.port, path="/mcp")


if __name__ == "__main__":
    main()
