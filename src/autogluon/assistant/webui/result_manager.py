# src/autogluon/assistant/webui/result_manager.py

import json
import os
import re
import shutil
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st


class ResultManager:
    """Manages task results viewing and downloading"""

    def __init__(self, output_dir: str, run_id: str = None):
        self.output_dir = Path(output_dir)
        self.run_id = run_id

    def extract_output_dir(self, phase_states: Dict) -> Optional[str]:
        """Extract output directory from log phase states"""
        output_phase = phase_states.get("Output", {})
        logs = output_phase.get("logs", [])

        for log in reversed(logs):
            # Look for "output saved in" pattern and extract the path
            match = re.search(r"output saved in\s+([^\s]+)", log)
            if match:
                output_dir = match.group(1).strip()
                # Remove any trailing punctuation
                output_dir = output_dir.rstrip(".,;:")
                return output_dir
        return None

    def find_latest_model(self) -> Optional[Path]:
        """Find the latest model directory by timestamp"""
        model_dirs = []
        pattern = re.compile(r"model_(\d{8})_(\d{6})")

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
        for ext in [".csv", ".pq", ".parquet"]:
            results_file = self.output_dir / f"results{ext}"
            if results_file.exists():
                return results_file
        return None

    def find_token_usage_file(self) -> Optional[Path]:
        """Find token usage JSON file"""
        token_file = self.output_dir / "token_usage.json"
        return token_file if token_file.exists() else None

    def find_latest_generation_iter(self) -> Optional[Path]:
        """Find the latest generation_iter directory"""
        generation_dirs = []
        pattern = re.compile(r"generation_iter_(\d+)")

        for item in self.output_dir.iterdir():
            if item.is_dir() and pattern.match(item.name):
                match = pattern.match(item.name)
                iter_num = int(match.group(1))
                generation_dirs.append((iter_num, item))

        if generation_dirs:
            # Sort by iteration number and return the latest
            generation_dirs.sort(key=lambda x: x[0], reverse=True)
            return generation_dirs[0][1]
        return None

    def create_download_zip(self, include_items: List[str]) -> bytes:
        """Create a zip file with selected items"""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_file:
            with zipfile.ZipFile(tmp_file.name, "w", zipfile.ZIP_DEFLATED) as zipf:

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
            with open(tmp_file.name, "rb") as f:
                zip_data = f.read()

            # Clean up
            os.unlink(tmp_file.name)
            return zip_data

    def render_download_tab(self):
        """Render the download tab"""
        # Center the heading
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.markdown("### Download Options")

        # Check what's available
        has_model = self.find_latest_model() is not None
        has_results = self.find_results_file() is not None
        has_token_usage = self.find_token_usage_file() is not None

        # Selection options
        download_options = []

        # Center the checkboxes
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            # All option
            all_checked = st.checkbox("All", key=f"download_all_{self.output_dir}")
            if all_checked:
                download_options.append("all")
            st.caption("Includes all intermediate code, logs, models, results, and token usage statistics")

            # Individual options (disabled if "All" is selected)
            disabled = all_checked

            # Add spacing between checkboxes
            st.markdown("")

            # Model option
            if has_model:
                if st.checkbox("Final trained model", disabled=disabled, key=f"download_model_{self.output_dir}"):
                    if not disabled:
                        download_options.append("model")
                model_dir = self.find_latest_model()
                st.caption(f"Latest model: {model_dir.name}")
                st.markdown("")

            # Results option
            if has_results:
                if st.checkbox("Results", disabled=disabled, key=f"download_results_{self.output_dir}"):
                    if not disabled:
                        download_options.append("results")
                results_file = self.find_results_file()
                st.caption(f"Results file: {results_file.name}")
                st.markdown("")

            # Token usage option
            if has_token_usage:
                if st.checkbox("Token usage", disabled=disabled, key=f"download_token_{self.output_dir}"):
                    if not disabled:
                        download_options.append("token_usage")
                st.caption("Token usage statistics (JSON)")

        # Download button - centered
        st.markdown("")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if download_options:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"autogluon_results_{self.output_dir.name}_{timestamp}.zip"

                # Create download directly
                zip_data = self.create_download_zip(download_options)

                st.download_button(
                    label="Download Package",
                    data=zip_data,
                    file_name=filename,
                    mime="application/zip",
                    key=f"download_btn_{self.output_dir}",
                    use_container_width=True,
                )
            else:
                st.info("Select items to download")

    def render_results_tab(self):
        """Render the results viewing tab"""
        results_file = self.find_results_file()

        if not results_file:
            st.info("No results file found in the output directory.")
            return

        st.markdown("### Results Visualization")

        try:
            # Load results based on file type
            if results_file.suffix == ".csv":
                df = pd.read_csv(results_file)
            else:  # .pq or .parquet
                df = pd.read_parquet(results_file)

            # Display basic info
            st.markdown(f"**File:** {results_file.name}")
            st.markdown(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")

            # Display the dataframe (all data)
            st.dataframe(df, use_container_width=True)

            # Show summary statistics for numeric columns
            numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
            if len(numeric_cols) > 0:
                st.markdown("**Summary Statistics:**")
                st.dataframe(df[numeric_cols].describe(), use_container_width=True)

        except Exception as e:
            st.error(f"Error loading results: {str(e)}")

    def render_code_tab(self):
        """Render the code viewing tab"""
        latest_gen_dir = self.find_latest_generation_iter()

        if not latest_gen_dir:
            st.info("No generation directories found in the output directory.")
            return

        st.markdown("### Generated Code")

        # Check for execution script
        exec_script = latest_gen_dir / "execution_script.sh"
        if exec_script.exists():
            st.markdown("**Execution Script for the Python code below:**")
            with open(exec_script, "r") as f:
                exec_code = f.read()
            st.code(exec_code, language="bash")

        # Check for generated Python code
        gen_code = latest_gen_dir / "generated_code.py"
        if gen_code.exists():
            st.markdown("**Generated Python Code:**")
            with open(gen_code, "r") as f:
                python_code = f.read()
            st.code(python_code, language="python")

        if not exec_script.exists() and not gen_code.exists():
            st.info("No code files found in the latest generation directory.")

    def find_input_dir(self) -> Optional[Path]:
        """Find the input directory associated with this output"""
        # Try to find from the output directory structure
        # Usually input directories are named upload_XXXXX in the user data directory
        from autogluon.assistant.webui.utils.utils import get_user_data_dir

        user_dir = get_user_data_dir()

        # Look for directories that might be associated with this task
        # This is a heuristic - you might need to adjust based on your actual directory structure
        for item in user_dir.iterdir():
            if item.is_dir() and item.name.startswith("upload_"):
                # Check if this upload directory has any connection to our output
                # This might require additional metadata or naming conventions
                # For now, we'll return None and let the caller handle it
                pass

        return None

    def delete_task_data(self) -> Tuple[bool, str]:
        """Delete both input and output directories for this task"""
        try:
            # Delete output directory
            if self.output_dir.exists():
                shutil.rmtree(self.output_dir)

            # Try to find and delete input directory
            # Note: This requires proper tracking of input directories per task
            # For now, we only delete the output directory

            return True, "Task data has been successfully deleted."
        except Exception as e:
            return False, f"Error deleting task data: {str(e)}"

    def render_feedback_tab(self):
        """Render the feedback and privacy tab"""
        st.markdown("### Feedback & Privacy")

        # Feedback form
        with st.form(key=f"feedback_form_{self.output_dir}"):
            # Star rating
            rating = st.feedback("stars", key=f"rating_{self.output_dir}")

            # Text feedback
            feedback_text = st.text_area(
                "Additional comments (optional):",
                placeholder="Share your thoughts about the results...",
                key=f"feedback_text_{self.output_dir}",
            )

            # Submit button
            submitted = st.form_submit_button("Submit Feedback", use_container_width=True)

            if submitted:
                # Check if at least one field is filled
                if rating is not None or feedback_text.strip():
                    # Create feedback directory
                    feedback_dir = self.output_dir / "feedback"
                    feedback_dir.mkdir(exist_ok=True)

                    # Create feedback data
                    feedback_data = {
                        "timestamp": datetime.now().isoformat(),
                        "rating": rating,
                        "comments": feedback_text.strip(),
                    }

                    # Save feedback
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    feedback_file = feedback_dir / f"feedback_{timestamp}.json"

                    with open(feedback_file, "w") as f:
                        json.dump(feedback_data, f, indent=2)

                    st.success("Thank you for your feedback! It has been saved.")
                    st.balloons()
                else:
                    st.warning("Please provide a rating or comments before submitting.")

        # Privacy notice
        st.markdown("---")
        st.markdown("**Privacy Notice:**")
        st.markdown(
            "Your feedback is stored locally in the output directory and is not automatically "
            "shared with anyone. You can choose to share the feedback file with the AutoGluon "
            "team if you wish to help improve the product."
        )

        # Delete task section
        st.markdown("---")
        st.markdown("**Delete Task Data:**")
        st.markdown(
            "If you want to remove this task and all associated data from your system, "
            "you can delete it here. This action cannot be undone."
        )

        # Create unique key for the dialog
        dialog_key = f"delete_dialog_{self.run_id or self.output_dir.name}"

        @st.dialog("Confirm Deletion")
        def confirm_delete():
            st.warning(
                "⚠️ **Warning:** This will permanently delete:\n"
                "- All output files (models, results, logs)\n"
                "- The associated input data\n"
                "- This task from your history\n\n"
                "This action cannot be undone!"
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Cancel", use_container_width=True, key=f"cancel_{dialog_key}"):
                    st.rerun()
            with col2:
                if st.button("Delete", type="primary", use_container_width=True, key=f"confirm_{dialog_key}"):
                    # Set deletion flag in session state
                    st.session_state[f"delete_task_{self.run_id}"] = True
                    st.rerun()

        # Delete button
        if st.button("🗑️ Delete This Task", type="secondary", key=f"delete_btn_{self.output_dir}"):
            confirm_delete()

    def render(self):
        """Main render method for result manager"""
        if not self.output_dir.exists():
            st.error(f"Output directory not found: {self.output_dir}")
            return

        # Create tabs
        tabs = st.tabs(["Download", "See Results", "See Code", "Feedback & Privacy"])

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
        match = re.search(r"output saved in\s+([^\s]+)", log)
        if match:
            output_dir = match.group(1).strip()
            # Remove any trailing punctuation
            output_dir = output_dir.rstrip(".,;:")
            break

    if output_dir:
        # Store output dir in session state for this task
        task_output_key = f"task_output_{run_id}"
        st.session_state[task_output_key] = output_dir

        # Render result manager
        manager = ResultManager(output_dir, run_id)
        manager.render()
    else:
        st.warning("Output directory not found in logs. Results may not be available yet.")
