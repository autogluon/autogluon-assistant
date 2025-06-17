# src/autogluon/assistant/webui/backend/routes.py

import uuid

from flask import Blueprint, jsonify, request

from .utils import cancel_run, get_logs, get_status, parse_log_line, send_user_input, start_run

bp = Blueprint("api", __name__)


@bp.route("/run", methods=["POST"])
def run():
    """
    Receive frontend startup request with parameters similar to original mlzero CLI.
    Returns JSON { "run_id": "<uuid>" }.
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
        "-n",
        str(max_iter),
        "-v",
        str(verbosity),
        "-c",
        config_path,
    ]
    if out_dir:
        cmd += ["-o", out_dir]
    if init_prompt:
        cmd += ["-u", init_prompt]
    if control:
        cmd += ["--need-user-input"]
    if extract_dir:
        cmd += ["-e", extract_dir]

    run_id = uuid.uuid4().hex
    # Get credentials from request (now supports multiple providers)
    credentials = data.get("aws_credentials")  # Keep field name for backward compatibility
    start_run(run_id, cmd, credentials)
    return jsonify({"run_id": run_id})


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
    return jsonify(get_status(run_id))


@bp.route("/cancel", methods=["POST"])
def cancel():
    """
    Receive {"run_id": "..."} and terminate the run.
    """
    run_id = request.get_json().get("run_id", "")
    cancel_run(run_id)
    return jsonify({"cancelled": True})


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
