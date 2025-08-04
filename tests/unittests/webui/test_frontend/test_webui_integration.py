import tempfile
from pathlib import Path

from streamlit.testing.v1 import AppTest


class TestWebUIIntegration:
    """WebUI integration test - focuses on UI rendering and message display"""

    def test_webui_ui_rendering(self):
        """Test WebUI can render task history and results correctly"""

        # 1. Initialize Streamlit AppTest
        at = AppTest.from_file("src/autogluon/assistant/webui/Launch_MLZero.py")
        at.run()

        # 2. Verify initial welcome message is displayed
        found_message = False
        for msg in at.chat_message:
            for el in msg.markdown:
                if "Make sure your credentials are set in the" in str(el.value):
                    found_message = True
                    break
            if found_message:
                break
        assert found_message, "Welcome message not found"

        # 3. Test sidebar configuration controls
        # Verify checkbox for credentials exists and can be checked
        bedrock_checkbox = at.checkbox(key="bedrock_already_setup")
        assert bedrock_checkbox is not None, "Bedrock credentials checkbox not found"
        bedrock_checkbox.check()

        # Verify log verbosity slider exists and can be changed
        log_slider = at.select_slider(key="log_verbosity")
        assert log_slider is not None, "Log verbosity slider not found"
        log_slider.set_value("INFO")

        # Verify max iterations input exists and can be changed
        max_iter_input = at.number_input(key="max_iterations")
        assert max_iter_input is not None, "Max iterations input not found"
        max_iter_input.set_value(2)

        # Apply configuration changes
        at.run()

        # 4. Simulate a completed task in history
        # Import required classes
        from autogluon.assistant.webui.Launch_MLZero import Message

        # Create temporary output directory with mock results
        with tempfile.TemporaryDirectory() as temp_dir:
            test_output_dir = Path(temp_dir) / "test_output"
            test_output_dir.mkdir(parents=True, exist_ok=True)

            # Create mock result files that ResultManager expects
            (test_output_dir / "results.csv").write_text("id,prediction\n1,0\n2,1\n")
            (test_output_dir / "token_usage.json").write_text('{"total": {"total_tokens": 1000}}')

            # Create mock code generation files
            gen_iter_dir = test_output_dir / "generation_iter_1"
            gen_iter_dir.mkdir()
            (gen_iter_dir / "generated_code.py").write_text("# Test code\nprint('Hello')")
            (gen_iter_dir / "execution_script.sh").write_text("#!/bin/bash\npython generated_code.py")

            # Add a completed task to message history
            test_run_id = "test_run_123"
            test_input_dir = "/tmp/test_input_123"

            # Add user task submission message
            at.session_state["messages"].append(Message.user_summary("time limit: 5 mins", input_dir=test_input_dir))

            # Add task completion message with all execution phases
            phase_states = {
                "Reading": {
                    "status": "complete",
                    "logs": ["DataPerceptionAgent: beginning to scan data folder and group similar files."],
                },
                "Iteration 0": {
                    "status": "complete",
                    "logs": ["RetrieverAgent: generating search query and retrieving tutorials."],
                },
                "Output": {
                    "status": "complete",
                    "logs": ["Task completed successfully", f"output saved in {test_output_dir}"],
                },
            }

            at.session_state["messages"].append(
                Message.task_log(test_run_id, phase_states, 2, str(test_output_dir), test_input_dir)  # max_iter
            )

            # Add task results message to trigger ResultManager
            at.session_state["messages"].append(Message.task_results(test_run_id, str(test_output_dir)))

            # Render the updated messages
            at.run(timeout=10)

            # 5. Verify task log expanders are displayed correctly
            expanders = at.expander

            # Verify Reading phase expander exists
            reading_expander = next((e for e in expanders if "Reading" in str(e.label)), None)
            assert reading_expander is not None, "Reading expander not found in task history"

            # Verify Iteration 0 expander exists
            iteration_expander = next((e for e in expanders if "Iteration 0" in str(e.label)), None)
            assert iteration_expander is not None, "Iteration 0 expander not found in task history"

            # 6. Verify result tabs are displayed
            tabs = at.tabs
            assert len(tabs) > 0, "No tabs found. Expected result tabs to be displayed."

            # Verify all expected tabs are present
            tab_labels = [str(tab.label) for tab in tabs]
            expected_tabs = ["Download", "See Results", "See Code", "Feedback & Privacy"]

            for expected in expected_tabs:
                assert any(
                    expected in label for label in tab_labels
                ), f"Tab '{expected}' not found. Available tabs: {tab_labels}"

            # 7. Verify Download tab contains "All" checkbox option
            all_checkboxes = at.checkbox
            all_checkbox = next((cb for cb in all_checkboxes if "All" in str(cb.label)), None)
            assert all_checkbox is not None, "All checkbox not found in Download tab"
