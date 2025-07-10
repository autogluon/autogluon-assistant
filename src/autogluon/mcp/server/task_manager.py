"""
Task manager for MCP server - manages AutoGluon tasks via Flask API
"""

import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import aiohttp
except ImportError:
    print("Warning: aiohttp not installed. Install with: pip install aiohttp")
    aiohttp = None


from ..constants import API_URL, VERBOSITY_MAP

logger = logging.getLogger(__name__)


class TaskManager:
    """Manages AutoGluon tasks through Flask backend API"""

    def __init__(self, api_url: str = API_URL):
        """
        Initialize task manager.

        Args:
            api_url: Base URL for Flask API
        """
        self.api_url = api_url
        self.current_task = None
        self.task_lock = threading.Lock()

    async def start_task(self, params: dict) -> dict:
        """
        Start a new AutoGluon task.

        Args:
            params: Task parameters including paths, config, credentials

        Returns:
            dict: Result with success status and task info
        """
        with self.task_lock:
            # Check if a task is already running
            if self.current_task and self.current_task.get("state") == "running":
                return {"success": False, "error": "A task is already running. Please wait or cancel it first."}

            try:
                # Prepare request payload
                payload = {
                    "data_src": params["input_dir"],
                    "config_path": params.get("config_path", ""),
                    "max_iter": params.get("max_iterations", 5),
                    "init_prompt": params.get("initial_user_input"),
                    "control": False,
                    "verbosity": VERBOSITY_MAP.get("INFO", "2"),
                    "out_dir": params["server_output_dir"],
                }

                # Add credentials based on provider
                if params.get("credentials"):
                    # Use the same field name as WebUI for compatibility
                    payload["aws_credentials"] = params["credentials"]

                # Make async request to Flask API
                async with aiohttp.ClientSession() as session:
                    async with session.post(f"{self.api_url}/run", json=payload) as response:
                        if response.status == 200:
                            result = await response.json()

                            # Store current task info
                            self.current_task = {
                                "task_id": result["task_id"],
                                "run_id": None,  # Will be set when task starts
                                "state": "queued",
                                "position": result.get("position", 0),
                                "started_at": datetime.now().isoformat(),
                                "params": params,
                                "logs": [],
                            }

                            logger.info(f"Task submitted: {result['task_id']}")

                            return {
                                "success": True,
                                "task_id": result["task_id"],
                                "position": result.get("position", 0),
                            }
                        else:
                            error_text = await response.text()
                            return {"success": False, "error": f"API error: {response.status} - {error_text}"}

            except Exception as e:
                logger.error(f"Failed to start task: {str(e)}")
                return {"success": False, "error": str(e)}

    async def check_status(self) -> dict:
        """
        Check status of current task.

        Returns:
            dict: Task status including state, logs, and progress
        """
        with self.task_lock:
            if not self.current_task:
                return {"success": True, "state": "idle", "message": "No task is currently running"}

            try:
                task_id = self.current_task["task_id"]

                # First check queue status if we don't have run_id yet
                if not self.current_task.get("run_id"):
                    async with aiohttp.ClientSession() as session:
                        async with session.get(f"{self.api_url}/queue/status/{task_id}") as response:
                            if response.status == 200:
                                queue_status = await response.json()

                                # Update task info
                                if queue_status.get("run_id"):
                                    self.current_task["run_id"] = queue_status["run_id"]
                                    self.current_task["state"] = "running"
                                else:
                                    self.current_task["position"] = queue_status.get("position", 0)

                                return {
                                    "success": True,
                                    "state": self.current_task["state"],
                                    "position": self.current_task.get("position"),
                                    "task_id": task_id,
                                    "run_id": self.current_task.get("run_id"),
                                    "logs": [],
                                    "waiting_for_input": False,
                                }

                # If we have run_id, check actual task status
                run_id = self.current_task.get("run_id")
                if run_id:
                    # Get logs
                    logs_response = await self._fetch_logs(run_id)

                    # Get status
                    status_response = await self._fetch_status(run_id)

                    # Update current task
                    if logs_response["success"]:
                        self.current_task["logs"] = logs_response["logs"]

                    # Check if task finished
                    if status_response.get("finished", False):
                        self.current_task["state"] = "completed"
                        self.current_task["output_dir"] = status_response.get("output_dir")

                    return {
                        "success": True,
                        "state": self.current_task["state"],
                        "task_id": task_id,
                        "run_id": run_id,
                        "logs": self.current_task["logs"],
                        "waiting_for_input": status_response.get("waiting_for_input", False),
                        "input_prompt": status_response.get("input_prompt"),
                        "output_dir": self.current_task.get("output_dir"),
                    }
                else:
                    return {
                        "success": True,
                        "state": self.current_task["state"],
                        "position": self.current_task.get("position"),
                        "task_id": task_id,
                        "logs": [],
                        "waiting_for_input": False,
                    }

            except Exception as e:
                logger.error(f"Failed to check status: {str(e)}")
                return {"success": False, "error": str(e)}

    async def cancel_task(self) -> dict:
        """
        Cancel the currently running task.

        Returns:
            dict: Result of cancellation
        """
        with self.task_lock:
            if not self.current_task:
                return {"success": False, "error": "No task to cancel"}

            try:
                async with aiohttp.ClientSession() as session:
                    payload = {}

                    # Add appropriate ID based on task state
                    if self.current_task.get("run_id"):
                        payload["run_id"] = self.current_task["run_id"]
                    else:
                        payload["task_id"] = self.current_task["task_id"]

                    async with session.post(f"{self.api_url}/cancel", json=payload) as response:
                        if response.status == 200:
                            result = await response.json()

                            if result.get("cancelled", False):
                                self.current_task = None
                                return {"success": True}
                            else:
                                return {"success": False, "error": "Failed to cancel task"}
                        else:
                            return {"success": False, "error": f"API error: {response.status}"}

            except Exception as e:
                logger.error(f"Failed to cancel task: {str(e)}")
                return {"success": False, "error": str(e)}

    async def list_outputs(self) -> dict:
        """
        List output files from completed task.

        Returns:
            dict: List of output files with full paths and output directory
        """
        with self.task_lock:
            if not self.current_task:
                return {"success": False, "error": "No task found"}

            if self.current_task.get("state") != "completed":
                return {"success": False, "error": "Task is not completed yet"}

            output_dir = self.current_task.get("output_dir")
            if not output_dir:
                # Try to get from server output dir
                output_dir = self.current_task["params"].get("server_output_dir")

            if not output_dir:
                return {"success": False, "error": "Output directory not found"}

            try:
                # List files in output directory
                from ..file_handler import FileHandler

                file_handler = FileHandler(Path.home() / ".autogluon_assistant" / "mcp_uploads")
                files = file_handler.list_files(output_dir)

                # Return full paths
                full_paths = [str(Path(output_dir) / f) for f in files]

                return {"success": True, "files": full_paths, "output_dir": output_dir}  # Include the output directory

            except Exception as e:
                logger.error(f"Failed to list outputs: {str(e)}")
                return {"success": False, "error": str(e)}

    async def get_current_task_info(self) -> Optional[dict]:
        """Get information about current task"""
        with self.task_lock:
            if self.current_task:
                return {
                    "task_id": self.current_task.get("task_id"),
                    "run_id": self.current_task.get("run_id"),
                    "state": self.current_task.get("state"),
                    "started_at": self.current_task.get("started_at"),
                    "input_dir": self.current_task["params"].get("input_dir"),
                    "server_output_dir": self.current_task["params"].get("server_output_dir"),
                    "client_output_dir": self.current_task["params"].get("client_output_dir"),
                }
            return None

    async def get_progress(self) -> dict:
        """Get current task progress"""
        with self.task_lock:
            if not self.current_task:
                return {"progress": 0.0, "message": "No task running"}

            # Simple progress estimation based on logs
            logs = self.current_task.get("logs", [])
            max_iter = self.current_task["params"].get("max_iterations", 5)

            # Count iterations from logs
            current_iter = 0
            for log in logs:
                if isinstance(log, dict) and "Starting iteration" in log.get("text", ""):
                    current_iter += 1

            # Calculate progress
            if self.current_task["state"] == "completed":
                progress = 1.0
            elif self.current_task["state"] == "queued":
                progress = 0.0
            else:
                # Running - estimate based on iterations
                progress = min(current_iter / (max_iter + 2), 0.9)  # +2 for setup and finalization

            return {
                "progress": progress,
                "current_iteration": current_iter,
                "max_iterations": max_iter,
                "state": self.current_task["state"],
            }

    async def _fetch_logs(self, run_id: str) -> dict:
        """Fetch logs from Flask API"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_url}/logs", params={"run_id": run_id}) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {"success": True, "logs": result.get("lines", [])}
                    else:
                        return {"success": False, "logs": []}
        except Exception as e:
            logger.error(f"Failed to fetch logs: {str(e)}")
            return {"success": False, "logs": []}

    async def _fetch_status(self, run_id: str) -> dict:
        """Fetch status from Flask API"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_url}/status", params={"run_id": run_id}) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        return {"finished": False}
        except Exception as e:
            logger.error(f"Failed to fetch status: {str(e)}")
            return {"finished": False}
