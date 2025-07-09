# src/autogluon/assistant/webui/backend/utils.py

import logging
import os
import signal
import subprocess
import threading
import uuid
from typing import Dict, List, Optional

from autogluon.assistant.constants import WEBUI_INPUT_MARKER, WEBUI_INPUT_REQUEST, WEBUI_OUTPUT_DIR

# Setup logging - reduce verbosity
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Silence watchdog debug logs
logging.getLogger("watchdog").setLevel(logging.WARNING)

# Global storage for each run's state
_runs: Dict[str, Dict] = {}


def parse_log_line(line: str) -> dict:
    """
    Parse a log line according to format "<LEVEL> <content>".
    Also detect special WebUI input requests and output directory.

    Returns:
        {
            "level": "<BRIEF/INFO/MODEL_INFO or other>",
            "text": "<content text>",
            "special": "<type of special message if any>"
        }
    """
    # Skip empty lines
    if not line.strip():
        return None

    # Check for special WebUI output directory
    if line.strip().startswith(WEBUI_OUTPUT_DIR):
        output_dir = line.strip()[len(WEBUI_OUTPUT_DIR) :].strip()
        return {"level": "OUTPUT_DIR", "text": output_dir, "special": "output_dir"}

    # Check for special WebUI input request
    if line.strip().startswith(WEBUI_INPUT_REQUEST):
        prompt = line.strip()[len(WEBUI_INPUT_REQUEST) :].strip()
        return {"level": "INPUT_REQUEST", "text": prompt, "special": "input_request"}

    # Regular log parsing
    allowed_levels = {"ERROR", "BRIEF", "INFO", "DETAIL", "DEBUG", "WARNING"}
    stripped = line.strip()

    parts = stripped.split(" ", 1)
    if len(parts) == 2 and parts[0] in allowed_levels:
        # Skip empty BRIEF logs
        if parts[0] == "BRIEF" and not parts[1].strip():
            return None
        return {"level": parts[0], "text": parts[1]}
    else:
        return {"level": "other", "text": stripped}


def start_run(task_id: str, cmd: List[str], credentials: Optional[Dict[str, str]] = None) -> str:
    """
    Start subprocess with stdin/stdout/stderr pipes.
    Set AUTOGLUON_WEBUI environment variable to indicate WebUI environment.
    Optionally set credentials (AWS, OpenAI, Anthropic) if provided.

    Returns: run_id (which is generated here, not task_id)
    """
    # Generate unique run_id
    run_id = uuid.uuid4().hex

    _runs[run_id] = {
        "process": None,
        "logs": [],
        "pointer": 0,
        "finished": False,
        "waiting_for_input": False,
        "input_prompt": None,
        "output_dir": None,
        "lock": threading.Lock(),
        "task_id": task_id,  # Store task_id for reference
    }

    def _target():
        try:
            # Set environment variable to indicate WebUI
            env = os.environ.copy()
            env["AUTOGLUON_WEBUI"] = "true"

            # Set credentials if provided
            if credentials:
                logger.info(f"Setting credentials for task {task_id[:8]}...")

                # Apply all provided environment variables
                for key, value in credentials.items():
                    env[key] = value
                    # Log environment variables (mask sensitive values)
                    if "KEY" in key or "TOKEN" in key:
                        masked_value = value[:4] + "..." if len(value) > 4 else "***"
                        logger.info(f"Task {task_id[:8]}: Setting {key}={masked_value}")
                    else:
                        logger.info(f"Task {task_id[:8]}: Setting {key}={value}")

                # Log which type of credentials were set based on what's actually present
                if "AWS_ACCESS_KEY_ID" in credentials:
                    logger.info(f"Task {task_id[:8]}: AWS credentials configured")
                if "OPENAI_API_KEY" in credentials:
                    logger.info(f"Task {task_id[:8]}: OpenAI API key configured")
                if "ANTHROPIC_API_KEY" in credentials:
                    logger.info(f"Task {task_id[:8]}: Anthropic API key configured")
            else:
                logger.info(f"Task {task_id[:8]}: No credentials provided, using system defaults")

            # Log the command being executed for debugging
            logger.info(f"Task {task_id[:8]}: Executing command: {' '.join(cmd)}")

            # Create process with stdin pipe
            p = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.PIPE,  # Enable stdin pipe
                text=True,
                env=env,
                bufsize=1,  # Line buffered
                # Create new process group for proper termination
                preexec_fn=os.setsid if os.name != "nt" else None,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
            )
            _runs[run_id]["process"] = p

            logger.info(f"Started task {task_id[:8]} with run_id {run_id[:8]}...")

            # Read stdout line by line
            for line in p.stdout:
                line = line.rstrip("\n")

                with _runs[run_id]["lock"]:
                    # Parse the line
                    parsed = parse_log_line(line)

                    # Skip None results (empty lines, etc.)
                    if parsed is None:
                        continue

                    # Check if this is output directory notification
                    if parsed.get("special") == "output_dir":
                        _runs[run_id]["output_dir"] = parsed["text"]
                        logger.info(f"Task {task_id[:8]} output directory: {parsed['text']}")
                        # Don't add this to logs
                        continue

                    # Check if this is an input request
                    if parsed.get("special") == "input_request":
                        _runs[run_id]["waiting_for_input"] = True
                        _runs[run_id]["input_prompt"] = parsed["text"]
                        logger.info(f"Task {task_id[:8]} requesting user input")

                    # Always append to logs (original line, not parsed)
                    _runs[run_id]["logs"].append(line)

            p.wait()
            exit_code = p.returncode
            logger.info(f"Task {task_id[:8]} completed with exit code {exit_code}")

        except Exception as e:
            logger.error(f"Error in task {task_id[:8]}: {str(e)}", exc_info=True)
            with _runs[run_id]["lock"]:
                _runs[run_id]["logs"].append(f"Process error: {str(e)}")
        finally:
            with _runs[run_id]["lock"]:
                _runs[run_id]["finished"] = True
                _runs[run_id]["waiting_for_input"] = False

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()

    return run_id


def send_user_input(run_id: str, user_input: str) -> bool:
    """
    Send user input to the subprocess stdin.
    Returns True if successful, False otherwise.
    """
    info = _runs.get(run_id)
    if not info:
        logger.error(f"Run {run_id} not found")
        return False

    with info["lock"]:
        process = info.get("process")
        if not process or not process.stdin or process.poll() is not None:
            logger.error(f"Process not available for input: {run_id}")
            return False

        try:
            # Send input with special marker
            input_line = f"{WEBUI_INPUT_MARKER}{user_input}\n"
            process.stdin.write(input_line)
            process.stdin.flush()

            # Reset input waiting state
            info["waiting_for_input"] = False
            info["input_prompt"] = None

            # Log the user input for display with proper formatting
            if user_input:
                info["logs"].append(f"BRIEF User input: {user_input}")
            else:
                info["logs"].append("BRIEF User input: (skipped)")

            logger.info(f"Sent input to task {run_id[:8]}")
            return True

        except Exception as e:
            logger.error(f"Error sending input to task {run_id[:8]}: {str(e)}")
            return False


def get_logs(run_id: str) -> List[str]:
    """
    Return list of new log lines since last call.
    """
    info = _runs.get(run_id)
    if info is None:
        return []

    with info["lock"]:
        logs = info["logs"]
        ptr = info["pointer"]
        new = logs[ptr:]
        info["pointer"] = len(logs)
        return new


def get_status(run_id: str) -> dict:
    """
    Return task status including whether it's waiting for input and output directory.
    """
    info = _runs.get(run_id)
    if info is None:
        return {"finished": True, "error": "run_id not found"}

    with info["lock"]:
        return {
            "finished": info["finished"],
            "waiting_for_input": info.get("waiting_for_input", False),
            "input_prompt": info.get("input_prompt", None),
            "output_dir": info.get("output_dir", None),
        }


def cancel_run(run_id: str):
    """
    Terminate the subprocess for the given run.
    """
    info = _runs.get(run_id)
    if info and info["process"] and not info["finished"]:
        process = info["process"]

        try:
            if os.name == "nt":  # Windows
                process.terminate()
            else:  # Unix/Linux/Mac
                # Send SIGTERM to entire process group
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)

            # Give process time to exit gracefully
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if needed
                if os.name == "nt":
                    process.kill()
                else:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                process.wait()

            # Add cancellation log
            with info["lock"]:
                info["logs"].append("Task cancelled by user")

        except Exception as e:
            with info["lock"]:
                info["logs"].append(f"Error cancelling task: {str(e)}")
        finally:
            with info["lock"]:
                info["finished"] = True
                info["waiting_for_input"] = False
