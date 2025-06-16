# src/autogluon/assistant/webui/log_processor.py

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import streamlit as st


# Phase matching configuration
@dataclass
class PhasePatterns:
    """Log phase matching patterns"""

    READING_START = "DataPerceptionAgent: beginning to scan data folder and group similar files."
    READING_END = "ToolSelectorAgent: selected"
    ITER_START = re.compile(r"Starting iteration (\d+)!")
    ITER_END = re.compile(r"Code generation (failed|successful)")
    OUTPUT_START = "Total tokens"
    OUTPUT_END = "output saved in"


@dataclass
class PhaseInfo:
    """Phase information"""

    status: str = "running"  # running or complete
    logs: List[str] = field(default_factory=list)


class LogProcessor:
    """Log processor - create an independent instance for each task"""

    def __init__(self, max_iter: int):
        self.max_iter = max_iter
        self.patterns = PhasePatterns()
        self.current_phase: Optional[str] = None
        self.phase_states: Dict[str, PhaseInfo] = {}
        self.processed_count = 0
        self.waiting_for_input = False
        self.input_prompt = None
        self.output_dir = None

    @property
    def progress(self) -> float:
        """Calculate current progress"""
        total_stages = self.max_iter + 2

        # Progress of current phase
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

        # Calculate based on completed phases
        completed = sum(1 for phase in self.phase_states.values() if phase.status == "complete")
        return min(completed / total_stages, 1.0)

    def process_new_logs(self, log_entries: List[Dict]) -> None:
        """Process new log entries"""
        # Only process new logs
        new_entries = log_entries[self.processed_count :]

        for entry in new_entries:
            level = entry.get("level", "")
            text = entry.get("text", "")
            special = entry.get("special", "")

            # Handle special messages
            if special == "output_dir":
                self.output_dir = text
                print(f"DEBUG LogProcessor: Got output_dir = {text}")
                # Don't add to regular logs
                continue
            elif special == "input_request":
                self.waiting_for_input = True
                self.input_prompt = text
                print("DEBUG LogProcessor: Got input request, waiting_for_input = True")
                # Don't add input requests to regular logs
                continue

            # Skip empty BRIEF logs
            if level == "BRIEF" and not text.strip():
                continue

            # Process the log entry
            self._process_log_entry(text)

        self.processed_count = len(log_entries)

    def _clean_markup(self, text: str) -> str:
        """Remove rich text markup tags from log text"""
        # Remove [bold green], [bold red], [/bold green], [/bold red] tags
        cleaned = re.sub(r"\[/?bold\s*(green|red)\]", "", text)
        return cleaned

    def _process_log_entry(self, text: str) -> None:
        """Process a single log entry"""
        # Clean markup from text before processing
        clean_text = self._clean_markup(text)

        # Detect phase changes
        phase_change = self._detect_phase_change(text)

        if phase_change:
            phase_name, action = phase_change

            if action == "start":
                self.current_phase = phase_name
                if phase_name not in self.phase_states:
                    self.phase_states[phase_name] = PhaseInfo()
                self.phase_states[phase_name].logs.append(clean_text)

            elif action == "end":
                if phase_name in self.phase_states:
                    self.phase_states[phase_name].status = "complete"
                    self.phase_states[phase_name].logs.append(clean_text)
                self.current_phase = None
        else:
            # Add to current phase
            if self.current_phase and self.current_phase in self.phase_states:
                self.phase_states[self.current_phase].logs.append(clean_text)

    def _detect_phase_change(self, text: str) -> Optional[Tuple[str, str]]:
        """Detect phase changes"""
        # Reading phase
        if self.patterns.READING_START in text and "Reading" not in self.phase_states:
            return ("Reading", "start")
        elif self.patterns.READING_END in text and self.current_phase == "Reading":
            return ("Reading", "end")

        # Iteration phase
        m_start = self.patterns.ITER_START.search(text)
        if m_start:
            phase_name = f"Iteration {m_start.group(1)}"
            if phase_name not in self.phase_states:
                return (phase_name, "start")

        if self.patterns.ITER_END.search(text) and self.current_phase and self.current_phase.startswith("Iteration"):
            return (self.current_phase, "end")

        # Output phase
        if self.patterns.OUTPUT_START in text and "Output" not in self.phase_states:
            return ("Output", "start")
        elif self.patterns.OUTPUT_END in text and self.current_phase == "Output":
            return ("Output", "end")

        return None

    def render(self, show_progress: bool = True) -> None:
        """Render log UI"""
        if show_progress:
            if self.waiting_for_input and self.input_prompt:
                # Show input request prominently
                st.info(f"💬 {self.input_prompt}")
            elif self.current_phase:
                st.markdown(f"### {self.current_phase}")
            st.progress(self.progress)

        # Render each phase
        phase_order = ["Reading"] + [f"Iteration {i}" for i in range(self.max_iter)] + ["Output"]

        for phase_name in phase_order:
            if phase_name in self.phase_states:
                phase_info = self.phase_states[phase_name]
                is_expanded = show_progress and (phase_name == self.current_phase)

                with st.expander(phase_name, expanded=is_expanded):
                    for log in phase_info.logs:
                        # Logs are already cleaned in _process_log_entry
                        st.write(log)


# Convenience functions (maintain backward compatibility)


def process_logs(log_entries: List[Dict], max_iter: int) -> Dict:
    """Process complete logs and return structured data (for completed tasks)"""
    processor = LogProcessor(max_iter)
    processor.process_new_logs(log_entries)

    return {
        "phase_states": {
            name: {"status": info.status, "logs": info.logs} for name, info in processor.phase_states.items()
        },
        "progress": processor.progress,
        "current_phase": processor.current_phase,
    }


def render_task_logs(
    phase_states: Dict, max_iter: int, show_progress: bool = True, current_phase: str = None, progress: float = 0.0
) -> None:
    """Render task logs (for completed tasks)"""
    # Create temporary processor for rendering
    processor = LogProcessor(max_iter)

    # Restore state
    for phase_name, phase_data in phase_states.items():
        processor.phase_states[phase_name] = PhaseInfo(
            status=phase_data.get("status", "complete"), logs=phase_data.get("logs", [])
        )

    processor.current_phase = current_phase if show_progress else None
    processor.render(show_progress=show_progress)


def messages(log_entries: List[Dict], max_iter: int) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Process real-time logs (for running tasks)
    Returns: (waiting_for_input, input_prompt, output_dir)
    """
    run_id = st.session_state.get("run_id", "unknown")
    processor_key = f"log_processor_{run_id}"

    # Get or create processor
    if processor_key not in st.session_state:
        st.session_state[processor_key] = LogProcessor(max_iter)

    processor = st.session_state[processor_key]
    processor.process_new_logs(log_entries)
    processor.render(show_progress=True)

    return processor.waiting_for_input, processor.input_prompt, processor.output_dir
