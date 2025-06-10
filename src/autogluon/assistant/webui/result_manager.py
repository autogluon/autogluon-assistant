# src/autogluon/assistant/webui/result_manager.py

import os
import re
import json
import zipfile
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import streamlit as st


class ResultManager:
    """Manages task results viewing and downloading"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        # Debug: Print the path being used
        print(f"ResultManager initialized with output_dir: {output_dir}")
        print(f"Path object: {self.output_dir}")
        print(f"Absolute path: {self.output_dir.absolute()}")
        
    def extract_output_dir(self, phase_states: Dict) -> Optional[str]:
        """Extract output directory from log phase states"""
        output_phase = phase_states.get("Output", {})
        logs = output_phase.get("logs", [])
        
        # Look for output directory in the last log entry
        for log in reversed(logs):
            match = re.search(r'output saved in (.+?)(?:\s|$)', log)
            if match:
                return match.group(1)
        return None
    
    def find_latest_model(self) -> Optional[Path]:
        """Find the latest model directory by timestamp"""
        model_dirs = []
        pattern = re.compile(r'model_(\d{8})_(\d{6})')
        
        for item in self.output_dir.iterdir():
            if item.is_dir() and pattern.match(item.name):
                match = pattern.match(item.name)
                timestamp = match.group(1) + match.group(2)
                model_dirs.append((timestamp, item))
        
        if model_dirs:
            # Sort by timestamp and return the latest
            model_dirs.sort(key=lambda x: x[0], reverse=True)
            return model_dirs[0][1]
        return None
    
    def find_results_file(self) -> Optional[Path]:
        """Find results file (csv or parquet)"""
        for ext in ['.csv', '.pq', '.parquet']:
            results_file = self.output_dir / f"results{ext}"
            if results_file.exists():
                return results_file
        return None
    
    def find_token_usage_file(self) -> Optional[Path]:
        """Find token usage JSON file"""
        token_file = self.output_dir / "token_usage.json"
        return token_file if token_file.exists() else None
    
    def create_download_zip(self, include_items: List[str]) -> bytes:
        """Create a zip file with selected items"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
            with zipfile.ZipFile(tmp_file.name, 'w', zipfile.ZIP_DEFLATED) as zipf:
                
                if "all" in include_items:
                    # Add entire output directory
                    for root, dirs, files in os.walk(self.output_dir):
                        for file in files:
                            file_path = Path(root) / file
                            arcname = file_path.relative_to(self.output_dir.parent)
                            zipf.write(file_path, arcname)
                            
                else:
                    if "model" in include_items:
                        model_dir = self.find_latest_model()
                        if model_dir:
                            for root, dirs, files in os.walk(model_dir):
                                for file in files:
                                    file_path = Path(root) / file
                                    arcname = Path(self.output_dir.name) / file_path.relative_to(self.output_dir)
                                    zipf.write(file_path, arcname)
                    
                    if "results" in include_items:
                        results_file = self.find_results_file()
                        if results_file:
                            arcname = Path(self.output_dir.name) / results_file.name
                            zipf.write(results_file, arcname)
                    
                    if "token_usage" in include_items:
                        token_file = self.find_token_usage_file()
                        if token_file:
                            arcname = Path(self.output_dir.name) / token_file.name
                            zipf.write(token_file, arcname)
            
            # Read the zip file content
            with open(tmp_file.name, 'rb') as f:
                zip_data = f.read()
            
            # Clean up
            os.unlink(tmp_file.name)
            return zip_data
    
    def render_download_tab(self):
        """Render the download tab"""
        st.markdown("### 📥 Download Options")
        
        # Check what's available
        has_model = self.find_latest_model() is not None
        has_results = self.find_results_file() is not None
        has_token_usage = self.find_token_usage_file() is not None
        
        # Selection options
        download_options = []
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            if st.checkbox("All", key=f"download_all_{self.output_dir}"):
                download_options.append("all")
                
        with col2:
            st.caption("Includes all intermediate code, logs, models, results, and token usage statistics")
        
        # Individual options (disabled if "All" is selected)
        disabled = "all" in download_options
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if has_model and st.checkbox("Final trained model", disabled=disabled, 
                                        key=f"download_model_{self.output_dir}"):
                if not disabled:
                    download_options.append("model")
        with col2:
            if has_model:
                model_dir = self.find_latest_model()
                st.caption(f"Latest model: {model_dir.name}")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if has_results and st.checkbox("Results", disabled=disabled,
                                         key=f"download_results_{self.output_dir}"):
                if not disabled:
                    download_options.append("results")
        with col2:
            if has_results:
                results_file = self.find_results_file()
                st.caption(f"Results file: {results_file.name}")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if has_token_usage and st.checkbox("Token usage", disabled=disabled,
                                             key=f"download_token_{self.output_dir}"):
                if not disabled:
                    download_options.append("token_usage")
        with col2:
            if has_token_usage:
                st.caption("Token usage statistics (JSON)")
        
        # Download button
        if download_options:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"autogluon_results_{self.output_dir.name}_{timestamp}.zip"
            
            if st.button("🔽 Create Download", key=f"create_download_{self.output_dir}"):
                with st.spinner("Creating download package..."):
                    zip_data = self.create_download_zip(download_options)
                    
                st.download_button(
                    label="💾 Download Package",
                    data=zip_data,
                    file_name=filename,
                    mime="application/zip",
                    key=f"download_btn_{self.output_dir}"
                )
        else:
            st.info("Select items to download")
    
    def render_results_tab(self):
        """Render the results viewing tab"""
        st.info("📊 Results viewer - Coming soon!")
        # TODO: Implement results visualization
        # - Load results.csv/pq
        # - Show summary statistics
        # - Display performance metrics
        # - Show leaderboard if available
    
    def render_code_tab(self):
        """Render the code viewing tab"""
        st.info("👨‍💻 Code viewer - Coming soon!")
        # TODO: Implement code viewer
        # - Show final generated code
        # - Show execution scripts
        # - Syntax highlighting
        # - Copy to clipboard functionality
    
    def render_feedback_tab(self):
        """Render the feedback and privacy tab"""
        st.info("💬 Feedback & Privacy - Coming soon!")
        # TODO: Implement feedback system
        # - Rating system (1-5 stars)
        # - Text feedback
        # - Privacy settings
        # - Option to share results with AutoGluon team
    
    def render(self):
        """Main render method for result manager"""
        # Debug output
        if not self.output_dir.exists():
            st.error(f"Output directory not found: {self.output_dir}")
            # Try to see if parent directory exists
            if self.output_dir.parent.exists():
                st.info(f"Parent directory exists: {self.output_dir.parent}")
                # List directories in parent
                try:
                    dirs = [d for d in self.output_dir.parent.iterdir() if d.is_dir()]
                    if dirs:
                        st.info(f"Available directories: {[d.name for d in dirs[:5]]}")
                except Exception as e:
                    st.error(f"Error listing directories: {e}")
            return
        
        # Create tabs
        tabs = st.tabs(["📥 Download", "📊 See Results", "👨‍💻 See Code", "💬 Feedback & Privacy"])
        
        with tabs[0]:
            self.render_download_tab()
            
        with tabs[1]:
            self.render_results_tab()
            
        with tabs[2]:
            self.render_code_tab()
            
        with tabs[3]:
            self.render_feedback_tab()


def render_task_results(run_id: str, phase_states: Dict):
    """Convenience function to render task results"""
    # Extract output directory from phase states
    output_phase = phase_states.get("Output", {})
    logs = output_phase.get("logs", [])
    
    output_dir = None
    for log in reversed(logs):
        # Look for "output saved in" pattern and extract the path
        match = re.search(r'output saved in\s+([^\s]+)', log)
        if match:
            output_dir = match.group(1).strip()
            # Remove any trailing punctuation
            output_dir = output_dir.rstrip('.,;:')
            break
    
    if output_dir:
        # Store output dir in session state for this task
        task_output_key = f"task_output_{run_id}"
        st.session_state[task_output_key] = output_dir
        
        # Render result manager
        manager = ResultManager(output_dir)
        manager.render()
    else:
        st.warning("Output directory not found in logs. Results may not be available yet.")