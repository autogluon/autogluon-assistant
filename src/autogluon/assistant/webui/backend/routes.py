# src/autogluon/assistant/webui/backend/routes.py

import uuid
from flask import Blueprint, request, jsonify

from .utils import start_run, get_logs, get_status, cancel_run, parse_log_line, send_user_input

bp = Blueprint("api", __name__)

@bp.route("/run", methods=["POST"])
def run():
    """
    接收前端启动请求，参数同原 mlzero CLI。
    返回 JSON { "run_id": "<uuid>" }。
    """
    data = request.get_json()
    # 必要参数
    data_src    = data["data_src"]
    max_iter    = data["max_iter"]
    verbosity   = data["verbosity"]
    config_path = data["config_path"]
    # 可选
    out_dir     = data.get("out_dir")
    init_prompt = data.get("init_prompt")
    control     = data.get("control")
    extract_dir = data.get("extract_dir")
    aws_credentials = data.get("aws_credentials")  # 新增：AWS凭证

    # 构造命令行
    cmd = [
        "mlzero",
        "-i", data_src,
        "-n", str(max_iter),
        "-v", str(verbosity),
        "-c", config_path,
    ]
    if out_dir:     cmd += ["-o", out_dir]
    if init_prompt: cmd += ["-u", init_prompt]
    if control:     cmd += ["--need-user-input"]
    if extract_dir: cmd += ["-e", extract_dir]

    run_id = uuid.uuid4().hex
    start_run(run_id, cmd, aws_credentials)  # 传递AWS凭证
    return jsonify({"run_id": run_id})

@bp.route("/logs", methods=["GET"])
def logs():
    """
    返回指定 run_id 的新增日志行列表，每行是一个 JSON 对象：
      { "level": "...", "text": "...", "special": "..." }
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
    返回 {"finished": true/false, "waiting_for_input": true/false, "input_prompt": "..."}
    """
    run_id = request.args.get("run_id", "")
    return jsonify(get_status(run_id))

@bp.route("/cancel", methods=["POST"])
def cancel():
    """
    接收 {"run_id": "..."}，终止该 run。
    """
    run_id = request.get_json().get("run_id", "")
    cancel_run(run_id)
    return jsonify({"cancelled": True})

@bp.route("/input", methods=["POST"])
def send_input():
    """
    Send user input to a waiting process.
    接收 {"run_id": "...", "input": "..."}
    """
    data = request.get_json()
    run_id = data.get("run_id", "")
    user_input = data.get("input", "")
    
    success = send_user_input(run_id, user_input)
    return jsonify({"success": success})