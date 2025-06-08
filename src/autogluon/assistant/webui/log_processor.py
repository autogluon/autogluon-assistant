# src/autogluon/assistant/webui/log_processor.py

import re
import streamlit as st
from typing import Dict, List, Optional, Tuple
import hashlib

# ─── 阶段匹配关键字 ────────────────────────────────────────────
READING_START = "DataPerceptionAgent: beginning to scan data folder and group similar files."
READING_END = "ToolSelectorAgent: selected"
ITER_START_RE = re.compile(r"Starting iteration (\d+)!")
ITER_END_RE = re.compile(r"Code generation (failed|successful)")
OUTPUT_START = "Total tokens"
OUTPUT_END = "output saved in"


def process_logs(log_entries: List[Dict], max_iter: int) -> Dict:
    """处理日志并返回结构化的阶段数据（用于已完成的任务）
    
    Args:
        log_entries: 日志条目列表
        max_iter: 最大迭代次数
        
    Returns:
        包含阶段信息的字典
    """
    state = {
        "current_phase": None,
        "phase_states": {},
        "progress": 0.0,
        "stage_text": "",
    }
    
    total_stages = max_iter + 2
    
    # 处理所有日志
    for entry in log_entries:
        text = entry.get("text", "")
        
        # 检测阶段变化
        phase_change = _detect_phase_change(text, state)
        
        if phase_change:
            phase_name, action = phase_change
            
            if action == "start":
                state["current_phase"] = phase_name
                if phase_name not in state["phase_states"]:
                    state["phase_states"][phase_name] = {
                        "status": "running",
                        "logs": []
                    }
                state["phase_states"][phase_name]["logs"].append(text)
                state["stage_text"] = f"### {phase_name}"
                state["progress"] = _calculate_progress(state, total_stages, max_iter)
                
            elif action == "end":
                if phase_name in state["phase_states"]:
                    state["phase_states"][phase_name]["status"] = "complete"
                    state["phase_states"][phase_name]["logs"].append(text)
                state["current_phase"] = None
                if phase_name == "Output":
                    state["progress"] = 1.0
        else:
            # 普通日志，添加到当前阶段
            if state["current_phase"] and state["current_phase"] in state["phase_states"]:
                state["phase_states"][state["current_phase"]]["logs"].append(text)
    
    return state


def render_completed_task_logs(phase_states: Dict, max_iter: int):
    """渲染已完成任务的日志（用于历史记录）"""
    # 渲染所有阶段
    phase_order = ["Reading"] + [f"Iteration {i}" for i in range(max_iter)] + ["Output"]
    
    # 只渲染存在的阶段
    for phase_name in phase_order:
        if phase_name in phase_states:
            phase_data = phase_states[phase_name]
            
            # 创建 expander
            with st.expander(phase_name, expanded=False):
                for log in phase_data["logs"]:
                    st.write(log)


def render_task_logs(phase_states: Dict, max_iter: int, show_progress: bool = True, 
                     current_phase: str = None, progress: float = 0.0):
    """渲染任务日志UI
    
    Args:
        phase_states: 阶段状态字典
        max_iter: 最大迭代次数
        show_progress: 是否显示进度条
        current_phase: 当前运行的阶段
        progress: 当前进度
    """
    # 对于已完成的任务，使用简化版渲染
    if not show_progress:
        render_completed_task_logs(phase_states, max_iter)
        return
    
    # 显示进度条（仅运行中的任务）
    if current_phase:
        st.markdown(f"### {current_phase}")
    st.progress(progress)
    
    # 为当前任务创建隔离的渲染区域
    @st.fragment
    def render_current_task():
        # 渲染所有阶段
        phase_order = ["Reading"] + [f"Iteration {i}" for i in range(max_iter)] + ["Output"]
        
        # 只渲染存在的阶段
        for phase_name in phase_order:
            if phase_name in phase_states:
                phase_data = phase_states[phase_name]
                is_current = (phase_name == current_phase)
                
                # 创建 expander
                with st.expander(phase_name, expanded=is_current):
                    for log in phase_data["logs"]:
                        st.write(log)
    
    # 执行隔离的渲染
    render_current_task()


def messages(log_entries: List[Dict], max_iter: int):
    """处理实时日志并渲染UI（用于运行中的任务）"""
    # 获取当前运行的任务ID
    current_run_id = st.session_state.get("run_id", "unknown")
    
    # 为每个任务创建独立的状态key
    state_key = f"log_processor_state_{current_run_id}"
    
    # 初始化 session state
    if state_key not in st.session_state:
        st.session_state[state_key] = {
            "current_phase": None,
            "phase_states": {},
            "processed_logs": 0,
            "progress": 0.0,
            "stage_text": "",
            "fixed_max_iter": max_iter,
            "task_id": current_run_id,
        }
    
    state = st.session_state[state_key]
    fixed_max_iter = state["fixed_max_iter"]
    total_stages = fixed_max_iter + 2
    
    # 只处理属于当前任务的新日志
    if len(log_entries) > state["processed_logs"]:
        new_entries = log_entries[state["processed_logs"]:]
        
        for entry in new_entries:
            text = entry.get("text", "")
            
            # 检测阶段变化
            phase_change = _detect_phase_change(text, state)
            
            if phase_change:
                phase_name, action = phase_change
                
                if action == "start":
                    state["current_phase"] = phase_name
                    if phase_name not in state["phase_states"]:
                        state["phase_states"][phase_name] = {
                            "status": "running",
                            "logs": []
                        }
                    state["phase_states"][phase_name]["logs"].append(text)
                    state["stage_text"] = f"### {phase_name}"
                    state["progress"] = _calculate_progress(state, total_stages, fixed_max_iter)
                    
                elif action == "end":
                    if phase_name in state["phase_states"]:
                        state["phase_states"][phase_name]["status"] = "complete"
                        state["phase_states"][phase_name]["logs"].append(text)
                    state["current_phase"] = None
                    if phase_name == "Output":
                        state["progress"] = 1.0
            else:
                # 普通日志，添加到当前阶段
                if state["current_phase"] and state["current_phase"] in state["phase_states"]:
                    state["phase_states"][state["current_phase"]]["logs"].append(text)
        
        # 更新已处理的日志数量
        state["processed_logs"] = len(log_entries)
    
    # 渲染UI
    render_task_logs(
        state["phase_states"], 
        fixed_max_iter,
        show_progress=True,
        current_phase=state["current_phase"],
        progress=state["progress"]
    )


def _detect_phase_change(text: str, state: Dict) -> Optional[Tuple[str, str]]:
    """检测阶段变化，返回 (阶段名, 动作) 或 None"""
    # Reading 阶段
    if READING_START in text:
        # 只有当Reading还没开始时才返回start
        if "Reading" not in state.get("phase_states", {}):
            return ("Reading", "start")
    elif READING_END in text:
        # 只有当Reading存在且正在运行时才结束
        if state.get("current_phase") == "Reading":
            return ("Reading", "end")
    
    # Iteration 阶段
    m_start = ITER_START_RE.search(text)
    if m_start:
        idx = int(m_start.group(1))
        phase_name = f"Iteration {idx}"
        # 只有当这个迭代还没开始时才返回start
        if phase_name not in state.get("phase_states", {}):
            return (phase_name, "start")
    
    if ITER_END_RE.search(text):
        # 只结束当前正在运行的迭代
        if state.get("current_phase") and state["current_phase"].startswith("Iteration"):
            return (state["current_phase"], "end")
    
    # Output 阶段
    if OUTPUT_START in text:
        # 只有当Output还没开始时才返回start
        if "Output" not in state.get("phase_states", {}):
            return ("Output", "start")
    elif OUTPUT_END in text:
        # 只有当Output正在运行时才结束
        if state.get("current_phase") == "Output":
            return ("Output", "end")
    
    return None


def _calculate_progress(state: Dict, total_stages: int, max_iter: int) -> float:
    """计算当前进度"""
    if state["current_phase"] == "Reading":
        return 1.0 / total_stages
    elif state["current_phase"] == "Output":
        return (max_iter + 1) / total_stages
    elif state["current_phase"] and state["current_phase"].startswith("Iteration"):
        try:
            idx = int(state["current_phase"].split()[1])
            return (idx + 2) / total_stages
        except:
            pass
    
    # 根据已完成的阶段计算
    completed = 0
    if "Reading" in state["phase_states"] and state["phase_states"]["Reading"]["status"] == "complete":
        completed += 1
    
    for i in range(max_iter):
        phase_name = f"Iteration {i}"
        if phase_name in state["phase_states"] and state["phase_states"][phase_name]["status"] == "complete":
            completed += 1
    
    if "Output" in state["phase_states"] and state["phase_states"]["Output"]["status"] == "complete":
        completed += 1
    
    return min(completed / total_stages, 1.0)