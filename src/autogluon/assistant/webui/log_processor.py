# src/autogluon/assistant/webui/log_processor.py

import re
import streamlit as st
from typing import Dict, List, Optional, Tuple

# ─── 阶段匹配关键字 ────────────────────────────────────────────
READING_START = "DataPerceptionAgent: beginning to scan data folder and group similar files."
READING_END = "ToolSelectorAgent: selected"
ITER_START_RE = re.compile(r"Starting iteration (\d+)!")
ITER_END_RE = re.compile(r"Code generation (failed|successful)")
OUTPUT_START = "Total tokens"
OUTPUT_END = "output saved in"


def messages(log_entries: List[Dict], max_iter: int):
    """处理日志并渲染UI"""
    # 初始化 session state
    if "log_processor_state" not in st.session_state:
        st.session_state.log_processor_state = {
            "current_phase": None,
            "phase_states": {},  # {phase_name: {"status": "running/complete", "logs": []}}
            "processed_logs": 0,  # 已处理的日志数量
            "progress": 0.0,  # 当前进度
            "stage_text": "",  # 当前阶段文本
            "fixed_max_iter": max_iter,  # 固定的最大迭代次数
        }
    
    state = st.session_state.log_processor_state
    
    # 使用固定的 max_iter（任务开始时的值）
    fixed_max_iter = state["fixed_max_iter"]
    total_stages = fixed_max_iter + 2
    
    # 处理新日志（只处理未处理过的）
    new_entries = log_entries[state["processed_logs"]:]
    
    for i, entry in enumerate(new_entries):
        text = entry.get("text", "")
        log_index = state["processed_logs"] + i
        
        # 检测阶段变化
        phase_change = _detect_phase_change(text, state)
        
        if phase_change:
            phase_name, action = phase_change
            
            if action == "start":
                # 开始新阶段
                state["current_phase"] = phase_name
                if phase_name not in state["phase_states"]:
                    state["phase_states"][phase_name] = {
                        "status": "running",
                        "logs": []
                    }
                state["phase_states"][phase_name]["logs"].append(text)
                
                # 更新状态
                state["stage_text"] = f"### {phase_name}"
                state["progress"] = _calculate_progress(state, total_stages, fixed_max_iter)
                
            elif action == "end":
                # 结束阶段
                if phase_name in state["phase_states"]:
                    state["phase_states"][phase_name]["status"] = "complete"
                    state["phase_states"][phase_name]["logs"].append(text)
                
                state["current_phase"] = None
                
                # 如果是最后阶段，进度条满格
                if phase_name == "Output":
                    state["progress"] = 1.0
        else:
            # 普通日志，添加到当前阶段
            if state["current_phase"] and state["current_phase"] in state["phase_states"]:
                state["phase_states"][state["current_phase"]]["logs"].append(text)
    
    # 更新已处理的日志数量
    state["processed_logs"] = len(log_entries)
    
    # 基于 session state 渲染 UI（使用固定的 max_iter）
    _render_ui(state, fixed_max_iter)


def _render_ui(state: Dict, max_iter: int):
    """基于状态渲染 UI"""
    # 创建进度条和标题
    if state["stage_text"]:
        st.markdown(state["stage_text"])
    
    st.progress(state["progress"])
    
    # 渲染所有阶段（使用固定的 max_iter）
    phase_order = ["Reading"] + [f"Iteration {i}" for i in range(max_iter)] + ["Output"]
    
    # 只渲染已经出现在日志中的阶段
    for phase_name in phase_order:
        if phase_name in state["phase_states"]:
            phase_data = state["phase_states"][phase_name]
            is_current = (phase_name == state["current_phase"])
            
            # 创建 expander
            with st.expander(phase_name, expanded=is_current):
                for log in phase_data["logs"]:
                    st.write(log)


def _detect_phase_change(text: str, state: Dict) -> Optional[Tuple[str, str]]:
    """检测阶段变化，返回 (阶段名, 动作) 或 None"""
    # Reading 阶段
    if READING_START in text:
        return ("Reading", "start")
    elif READING_END in text:
        return ("Reading", "end")
    
    # Iteration 阶段
    m_start = ITER_START_RE.search(text)
    if m_start:
        idx = int(m_start.group(1))  # 这里直接使用日志中的索引，不需要加1
        return (f"Iteration {idx}", "start")
    
    if ITER_END_RE.search(text):
        # 如果当前在某个 Iteration 阶段，结束它
        if state["current_phase"] and state["current_phase"].startswith("Iteration"):
            return (state["current_phase"], "end")
        # 否则尝试找到一个未完成的 Iteration
        for phase_name, phase_data in state["phase_states"].items():
            if phase_name.startswith("Iteration") and phase_data["status"] == "running":
                return (phase_name, "end")
    
    # Output 阶段
    if OUTPUT_START in text:
        return ("Output", "start")
    elif OUTPUT_END in text:
        return ("Output", "end")
    
    return None


def _calculate_progress(state: Dict, total_stages: int, max_iter: int) -> float:
    """计算当前进度"""
    if state["current_phase"] == "Reading":
        return 1.0 / total_stages  # Reading 阶段的进度
    elif state["current_phase"] == "Output":
        return (max_iter + 1) / total_stages
    elif state["current_phase"] and state["current_phase"].startswith("Iteration"):
        try:
            idx = int(state["current_phase"].split()[1])
            # idx 从 0 开始，所以 idx=0 时进度应该是 2/total_stages
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