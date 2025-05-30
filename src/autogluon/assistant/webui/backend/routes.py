# src/autogluon/assistant/webui/backend/routes.py

import uuid
from flask import Blueprint, request, jsonify

from .utils import start_run, get_logs, get_status, cancel_run, parse_log_line

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

    # 构造命令行
    cmd = [
        "mlzero",
        "-i", data_src,
        "-n", str(max_iter),
        "-v", verbosity,
        "-c", config_path,
    ]
    if out_dir:     cmd += ["-o", out_dir]
    if init_prompt: cmd += ["-u", init_prompt]
    if control:     cmd += ["--need-user-input"]
    if extract_dir: cmd += ["-e", extract_dir]

    run_id = uuid.uuid4().hex
    start_run(run_id, cmd)
    return jsonify({"run_id": run_id})

@bp.route("/logs", methods=["GET"])
def logs():
    """
    返回指定 run_id 的新增日志行列表，每行是一个 JSON 对象：
      { "level": "...", "text": "..." }
    """
    run_id = request.args.get("run_id", "")
    raw_lines = get_logs(run_id)
    parsed = [parse_log_line(line) for line in raw_lines]
    return jsonify({"lines": parsed})

@bp.route("/status", methods=["GET"])
def status():
    """
    返回 {"finished": true/false}
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
