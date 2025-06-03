# src/autogluon/assistant/webui/backend/utils.py

import re
import subprocess
import threading


# 全局存储每个 run 的状态
_runs: dict = {}

def parse_log_line(line: str) -> dict:
    """
    按照“<LEVEL> <内容>”的格式解析一行日志，LEVEL 只能是 BRIEF、INFO、MODEL_INFO。
    如果满足格式，则提取出 level 和 text；否则返回 level="other"，text 为整行内容。

    返回：
        {
            "level": "<BRIEF/INFO/MODEL_INFO 或 other>",
            "text": "<内容文本>"
        }
    """
    allowed_levels = {"BRIEF", "INFO", "MODEL_INFO"}
    stripped = line.strip()

    parts = stripped.split(" ", 1)
    if len(parts) == 2 and parts[0] in allowed_levels:
        return {"level": parts[0], "text": parts[1]}
    else:
        return {"level": "other", "text": stripped}


# def parse_log_line(line: str) -> dict:
#     return line.strip()


def start_run(run_id: str, cmd: list[str]):
    """
    启动子进程，并在后台线程中持续读取 stdout/stderr，
    将每一行 append 到 _runs[run_id]['logs']。
    """
    _runs[run_id] = {
        "process": None,
        "logs": [],
        "pointer": 0,
        "finished": False,
    }

    def _target():
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        _runs[run_id]["process"] = p
        for line in p.stdout:
            _runs[run_id]["logs"].append(line.rstrip("\n"))
        p.wait()
        _runs[run_id]["finished"] = True

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()

def get_logs(run_id: str) -> list[str]:
    """
    返回自上次调用后新增的日志行列表。
    """
    info = _runs.get(run_id)
    if info is None:
        return []
    logs = info["logs"]
    ptr = info["pointer"]
    new = logs[ptr:]
    info["pointer"] = len(logs)
    return new

def get_status(run_id: str) -> dict:
    """
    返回任务是否完成。
    """
    info = _runs.get(run_id)
    if info is None:
        return {"finished": True, "error": "run_id not found"}
    return {"finished": info["finished"]}

def cancel_run(run_id: str):
    """
    终止对应 run 的子进程。
    """
    info = _runs.get(run_id)
    if info and info["process"]:
        info["process"].terminate()
        info["finished"] = True
