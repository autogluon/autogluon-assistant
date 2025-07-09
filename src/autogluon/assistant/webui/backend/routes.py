# src/autogluon/assistant/webui/backend/routes.py

import uuid

from flask import Blueprint, jsonify, request

from .queue import get_queue_manager
from .utils import cancel_run, get_logs, get_status, parse_log_line, send_user_input

bp = Blueprint("api", __name__)


@bp.route("/run", methods=["POST"])
def run():
    """
    Receive frontend startup request with parameters similar to original mlzero CLI.
    Returns JSON { "task_id": "<uuid>", "position": <int> }.
    """
    data = request.get_json()
    # Required parameters
    data_src = data["data_src"]
    max_iter = data["max_iter"]
    verbosity = data["verbosity"]
    config_path = data["config_path"]
    # Optional parameters
    out_dir = data.get("out_dir")
    init_prompt = data.get("init_prompt")
    control = data.get("control")
    extract_dir = data.get("extract_dir")

    # Build command line
    cmd = [
        "mlzero",
        "-i",
        data_src,
    ]
    if max_iter:
        cmd += ["--max-iterations", str(max_iter)]
    if verbosity:
        cmd += ["-v", str(verbosity)]
    if config_path:
        cmd += ["-c", config_path]
    if out_dir:
        cmd += ["-o", out_dir]
    if init_prompt:
        cmd += ["--initial-instruction", init_prompt]
    if control:
        cmd += ["--enable-per-iteration-instruction"]
    if extract_dir:
        cmd += ["-e", extract_dir]

    # Generate task_id
    task_id = uuid.uuid4().hex

    # Prepare command data
    command_data = {
        "cmd": cmd,
        "data_src": data_src,
        "config_path": config_path,
        "max_iter": max_iter,
        "control": control,
    }

    # Get credentials from request
    credentials = data.get("aws_credentials")  # Keep field name for backward compatibility

    # Submit to queue
    queue_manager = get_queue_manager()
    position = queue_manager.submit_task(task_id, command_data, credentials)

    return jsonify({"task_id": task_id, "position": position})


@bp.route("/logs", methods=["GET"])
def logs():
    """
    Return list of new log lines for specified run_id.
    Each line is a JSON object: { "level": "...", "text": "...", "special": "..." }
    """
    run_id = request.args.get("run_id", "")
    raw_lines = get_logs(run_id)
    # Filter out None values from parse_log_line
    parsed = [parse_log_line(line) for line in raw_lines]
    parsed = [p for p in parsed if p is not None]
    return jsonify({"lines": parsed})


@bp.route("/status", methods=["GET"])
def status():
    """
    Return {"finished": true/false, "waiting_for_input": true/false, "input_prompt": "..."}
    """
    run_id = request.args.get("run_id", "")
    status_info = get_status(run_id)

    # If task finished, notify queue manager
    if status_info.get("finished", False):
        queue_manager = get_queue_manager()
        queue_manager.complete_task_by_run_id(run_id)

    return jsonify(status_info)


@bp.route("/cancel", methods=["POST"])
def cancel():
    """
    Receive {"run_id": "..."} and terminate the run.
    Also supports {"task_id": "..."} for cancelling queued tasks.
    """
    data = request.get_json()
    run_id = data.get("run_id", "")
    task_id = data.get("task_id", "")

    if task_id:
        # Try to cancel queued task
        queue_manager = get_queue_manager()
        success = queue_manager.cancel_task(task_id)
        return jsonify({"cancelled": success})
    elif run_id:
        # Cancel running task
        cancel_run(run_id)
        # Also notify queue manager
        queue_manager = get_queue_manager()
        queue_manager.complete_task_by_run_id(run_id)
        return jsonify({"cancelled": True})
    else:
        return jsonify({"cancelled": False, "error": "No run_id or task_id provided"})


@bp.route("/input", methods=["POST"])
def send_input():
    """
    Send user input to a waiting process.
    Receive {"run_id": "...", "input": "..."}
    """
    data = request.get_json()
    run_id = data.get("run_id", "")
    user_input = data.get("input", "")

    success = send_user_input(run_id, user_input)
    return jsonify({"success": success})


@bp.route("/queue/status/<task_id>", methods=["GET"])
def queue_status(task_id):
    """Get status of a specific task in the queue"""
    queue_manager = get_queue_manager()
    status = queue_manager.get_task_status(task_id)

    if status is None:
        return jsonify({"error": "Task not found"}), 404

    return jsonify(status)


@bp.route("/queue/info", methods=["GET"])
def queue_info():
    """Get overall queue information"""
    queue_manager = get_queue_manager()
    info = queue_manager.get_queue_info()
    return jsonify(info)
