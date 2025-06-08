# src/autogluon/assistant/webui/log_processor.py

import re
import streamlit as st
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


# ─── 阶段匹配配置 ────────────────────────────────────────────
@dataclass
class PhasePatterns:
    """日志阶段匹配模式"""
    READING_START = "DataPerceptionAgent: beginning to scan data folder and group similar files."
    READING_END = "ToolSelectorAgent: selected"
    ITER_START = re.compile(r"Starting iteration (\d+)!")
    ITER_END = re.compile(r"Code generation (failed|successful)")
    OUTPUT_START = "Total tokens"
    OUTPUT_END = "output saved in"


@dataclass
class PhaseInfo:
    """阶段信息"""
    status: str = "running"  # running or complete
    logs: List[str] = field(default_factory=list)


class LogProcessor:
    """日志处理器 - 每个任务创建一个独立实例"""
    
    def __init__(self, max_iter: int):
        self.max_iter = max_iter
        self.patterns = PhasePatterns()
        self.current_phase: Optional[str] = None
        self.phase_states: Dict[str, PhaseInfo] = {}
        self.processed_count = 0
        
    @property
    def progress(self) -> float:
        """计算当前进度"""
        total_stages = self.max_iter + 2
        
        # 当前阶段的进度
        if self.current_phase == "Reading":
            return 1.0 / total_stages
        elif self.current_phase == "Output":
            return (self.max_iter + 1) / total_stages
        elif self.current_phase and self.current_phase.startswith("Iteration"):
            try:
                idx = int(self.current_phase.split()[1])
                return (idx + 2) / total_stages
            except:
                pass
        
        # 基于已完成阶段计算
        completed = sum(1 for phase in self.phase_states.values() 
                       if phase.status == "complete")
        return min(completed / total_stages, 1.0)
    
    def process_new_logs(self, log_entries: List[Dict]) -> None:
        """处理新的日志条目"""
        # 只处理新日志
        new_entries = log_entries[self.processed_count:]
        
        for entry in new_entries:
            text = entry.get("text", "")
            self._process_log_entry(text)
            
        self.processed_count = len(log_entries)
    
    def _process_log_entry(self, text: str) -> None:
        """处理单条日志"""
        # 检测阶段变化
        phase_change = self._detect_phase_change(text)
        
        if phase_change:
            phase_name, action = phase_change
            
            if action == "start":
                self.current_phase = phase_name
                if phase_name not in self.phase_states:
                    self.phase_states[phase_name] = PhaseInfo()
                self.phase_states[phase_name].logs.append(text)
                
            elif action == "end":
                if phase_name in self.phase_states:
                    self.phase_states[phase_name].status = "complete"
                    self.phase_states[phase_name].logs.append(text)
                self.current_phase = None
        else:
            # 添加到当前阶段
            if self.current_phase and self.current_phase in self.phase_states:
                self.phase_states[self.current_phase].logs.append(text)
    
    def _detect_phase_change(self, text: str) -> Optional[Tuple[str, str]]:
        """检测阶段变化"""
        # Reading 阶段
        if self.patterns.READING_START in text and "Reading" not in self.phase_states:
            return ("Reading", "start")
        elif self.patterns.READING_END in text and self.current_phase == "Reading":
            return ("Reading", "end")
        
        # Iteration 阶段
        m_start = self.patterns.ITER_START.search(text)
        if m_start:
            phase_name = f"Iteration {m_start.group(1)}"
            if phase_name not in self.phase_states:
                return (phase_name, "start")
        
        if self.patterns.ITER_END.search(text) and self.current_phase and self.current_phase.startswith("Iteration"):
            return (self.current_phase, "end")
        
        # Output 阶段
        if self.patterns.OUTPUT_START in text and "Output" not in self.phase_states:
            return ("Output", "start")
        elif self.patterns.OUTPUT_END in text and self.current_phase == "Output":
            return ("Output", "end")
        
        return None
    
    def render(self, show_progress: bool = True) -> None:
        """渲染日志UI"""
        if show_progress:
            if self.current_phase:
                st.markdown(f"### {self.current_phase}")
            st.progress(self.progress)
        
        # 渲染各阶段
        phase_order = ["Reading"] + [f"Iteration {i}" for i in range(self.max_iter)] + ["Output"]
        
        for phase_name in phase_order:
            if phase_name in self.phase_states:
                phase_info = self.phase_states[phase_name]
                is_expanded = show_progress and (phase_name == self.current_phase)
                
                with st.expander(phase_name, expanded=is_expanded):
                    for log in phase_info.logs:
                        st.write(log)


# ─── 便捷函数（保持向后兼容） ────────────────────────────────────

def process_logs(log_entries: List[Dict], max_iter: int) -> Dict:
    """处理完整的日志并返回结构化数据（用于已完成的任务）"""
    processor = LogProcessor(max_iter)
    processor.process_new_logs(log_entries)
    
    return {
        "phase_states": {name: {"status": info.status, "logs": info.logs} 
                        for name, info in processor.phase_states.items()},
        "progress": processor.progress,
        "current_phase": processor.current_phase,
    }


def render_task_logs(phase_states: Dict, max_iter: int, show_progress: bool = True, 
                     current_phase: str = None, progress: float = 0.0) -> None:
    """渲染任务日志（用于已完成的任务）"""
    # 创建临时处理器用于渲染
    processor = LogProcessor(max_iter)
    
    # 恢复状态
    for phase_name, phase_data in phase_states.items():
        processor.phase_states[phase_name] = PhaseInfo(
            status=phase_data.get("status", "complete"),
            logs=phase_data.get("logs", [])
        )
    
    processor.current_phase = current_phase if show_progress else None
    processor.render(show_progress=show_progress)


def messages(log_entries: List[Dict], max_iter: int) -> None:
    """处理实时日志（用于运行中的任务）"""
    run_id = st.session_state.get("run_id", "unknown")
    processor_key = f"log_processor_{run_id}"
    
    # 获取或创建处理器
    if processor_key not in st.session_state:
        st.session_state[processor_key] = LogProcessor(max_iter)
    
    processor = st.session_state[processor_key]
    processor.process_new_logs(log_entries)
    processor.render(show_progress=True)