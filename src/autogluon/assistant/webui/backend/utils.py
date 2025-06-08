# src/autogluon/assistant/webui/backend/utils.py

import re
import subprocess
import threading
import signal
import os


# 全局存储每个 run 的状态
_runs: dict = {}

def parse_log_line(line: str) -> dict:
    """
    按照"<LEVEL> <内容>"的格式解析一行日志，LEVEL 只能是 BRIEF、INFO、MODEL_INFO。
    如果满足格式，则提取出 level 和 text；否则返回 level="other"，text 为整行内容。

    返回：
        {
            "level": "<BRIEF/INFO/MODEL_INFO 或 other>",
            "text": "<内容文本>"
        }
    """
    allowed_levels = {"ERROR", "BRIEF", "INFO", "DETAIL", "DEBUG", "WARNING"}
    stripped = line.strip()

    parts = stripped.split(" ", 1)
    if len(parts) == 2 and parts[0] in allowed_levels:
        return {"level": parts[0], "text": parts[1]}
    else:
        return {"level": "other", "text": stripped}


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
        # 创建新的进程组，这样可以一次性终止整个进程树
        p = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True,
            # 在Unix系统上创建新的进程组
            preexec_fn=os.setsid if os.name != 'nt' else None,
            # 在Windows上使用CREATE_NEW_PROCESS_GROUP
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
        )
        _runs[run_id]["process"] = p
        
        try:
            for line in p.stdout:
                _runs[run_id]["logs"].append(line.rstrip("\n"))
            p.wait()
        except Exception as e:
            _runs[run_id]["logs"].append(f"Process error: {str(e)}")
        finally:
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
    if info and info["process"] and not info["finished"]:
        process = info["process"]
        
        try:
            if os.name == 'nt':  # Windows
                # 在Windows上使用terminate
                process.terminate()
            else:  # Unix/Linux/Mac
                # 发送SIGTERM到整个进程组，模拟Ctrl+C
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            
            # 给进程一些时间优雅地退出
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # 如果进程没有在5秒内退出，强制杀死
                if os.name == 'nt':
                    process.kill()
                else:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                process.wait()
            
            # 添加取消日志
            _runs[run_id]["logs"].append("Task cancelled by user")
            
        except Exception as e:
            _runs[run_id]["logs"].append(f"Error cancelling task: {str(e)}")
        finally:
            info["finished"] = True