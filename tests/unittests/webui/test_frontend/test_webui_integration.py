
import pytest


class TestWebUIIntegration:
    """WebUI integration test - focuses on core webui components"""

    def test_message_dataclass(self):
        """Test Message dataclass can create different message types"""
        from autogluon.assistant.webui.Launch_MLZero import Message

        # Test text message
        msg = Message.text("Hello world")
        assert msg.role == "assistant"
        assert msg.type == "text"
        assert msg.content["text"] == "Hello world"

        # Test user summary message
        msg = Message.user_summary("Test task", input_dir="/tmp/test")
        assert msg.role == "user"
        assert msg.type == "user_summary"
        assert msg.content["summary"] == "Test task"
        assert msg.content["input_dir"] == "/tmp/test"

        # Test task log message
        phase_states = {
            "Reading": {"status": "complete", "logs": ["Reading data..."]},
            "Iteration 1": {"status": "complete", "logs": ["Processing..."]},
        }
        msg = Message.task_log("run_123", phase_states, 2, "/tmp/output", "/tmp/input")
        assert msg.role == "assistant"
        assert msg.type == "task_log"
        assert msg.content["run_id"] == "run_123"
        assert msg.content["max_iter"] == 2
        assert msg.content["output_dir"] == "/tmp/output"
        assert msg.content["input_dir"] == "/tmp/input"

        # Test task results message
        msg = Message.task_results("run_456", "/tmp/results")
        assert msg.role == "assistant"
        assert msg.type == "task_results"
        assert msg.content["run_id"] == "run_456"
        assert msg.content["output_dir"] == "/tmp/results"

        # Test queue status message
        msg = Message.queue_status("task_789", 3)
        assert msg.role == "assistant"
        assert msg.type == "queue_status"
        assert msg.content["task_id"] == "task_789"
        assert msg.content["position"] == 3

    def test_log_processor(self):
        """Test LogProcessor can process log entries"""
        from autogluon.assistant.webui.log_processor import LogProcessor

        # Create processor for 2 iterations
        processor = LogProcessor(max_iter=2)

        # Test initial state
        assert processor.max_iter == 2
        assert processor.current_phase is None
        assert len(processor.phase_states) == 0
        assert processor.progress == 0.0

        # Process reading phase logs
        log_entries = [
            {"level": "INFO", "text": "DataPerceptionAgent: beginning to scan data folder and group similar files."},
            {"level": "INFO", "text": "Found 10 files"},
            {"level": "INFO", "text": "ToolSelectorAgent: selected python"},
        ]
        processor.process_new_logs(log_entries)

        # Verify reading phase was detected
        assert "Reading" in processor.phase_states
        assert processor.phase_states["Reading"].status == "complete"
        assert len(processor.phase_states["Reading"].logs) == 3

        # Process iteration logs
        iter_logs = [
            {"level": "INFO", "text": "Starting MCTS iteration 1"},
            {"level": "INFO", "text": "PythonCoderAgent: generating code"},
            {"level": "INFO", "text": "Node tree visualization generated at: /tmp/tree.png"},
        ]
        processor.process_new_logs(iter_logs)

        # Verify iteration phase
        assert "Iteration 1" in processor.phase_states
        assert processor.phase_states["Iteration 1"].status == "complete"

        # Process output phase
        output_logs = [
            {"level": "INFO", "text": "Total tokens used: 5000"},
            {"level": "INFO", "text": "Output saved in /tmp/output"},
        ]
        processor.process_new_logs(output_logs)

        # Verify output phase
        assert "Output" in processor.phase_states
        assert processor.phase_states["Output"].status == "complete"
        assert processor.progress == 1.0

    def test_log_processor_input_request(self):
        """Test LogProcessor handles input requests"""
        from autogluon.assistant.webui.log_processor import LogProcessor

        processor = LogProcessor(max_iter=1)

        # Process normal logs
        log_entries = [
            {"level": "INFO", "text": "Processing..."},
            {"level": "INFO", "text": "Waiting for input", "special": "input_request"},
        ]
        processor.process_new_logs(log_entries)

        # Verify input request was detected
        assert processor.waiting_for_input is True
        assert processor.input_prompt == "Waiting for input"

    def test_log_processor_output_dir(self):
        """Test LogProcessor handles output directory"""
        from autogluon.assistant.webui.log_processor import LogProcessor

        processor = LogProcessor(max_iter=1)

        # Process log with output directory
        log_entries = [
            {"level": "INFO", "text": "Starting..."},
            {"level": "OUTPUT_DIR", "text": "/tmp/test_output", "special": "output_dir"},
        ]
        processor.process_new_logs(log_entries)

        # Verify output directory was captured
        assert processor.output_dir == "/tmp/test_output"

    def test_log_processor_progress_calculation(self):
        """Test LogProcessor calculates progress correctly"""
        from autogluon.assistant.webui.log_processor import LogProcessor

        # Test with 2 iterations (total stages: Reading + 2 iterations + Output = 4)
        processor = LogProcessor(max_iter=2)

        # Complete reading phase
        processor.phase_states["Reading"] = type("PhaseInfo", (), {"status": "complete"})()
        processor.current_phase = None
        assert processor.progress == pytest.approx(0.25, abs=0.01)

        # Complete iteration 1
        processor.phase_states["Iteration 1"] = type("PhaseInfo", (), {"status": "complete"})()
        assert processor.progress == pytest.approx(0.5, abs=0.01)

        # Complete iteration 2
        processor.phase_states["Iteration 2"] = type("PhaseInfo", (), {"status": "complete"})()
        assert processor.progress == pytest.approx(0.75, abs=0.01)

        # Complete output
        processor.phase_states["Output"] = type("PhaseInfo", (), {"status": "complete"})()
        assert processor.progress == 1.0

    def test_process_logs_convenience_function(self):
        """Test process_logs convenience function"""
        from autogluon.assistant.webui.log_processor import process_logs

        log_entries = [
            {"level": "INFO", "text": "DataPerceptionAgent: beginning to scan data folder and group similar files."},
            {"level": "INFO", "text": "ToolSelectorAgent: selected python"},
            {"level": "INFO", "text": "Starting MCTS iteration 1"},
            {"level": "INFO", "text": "Node tree visualization generated at: /tmp/tree.png"},
        ]

        result = process_logs(log_entries, max_iter=1)

        # Verify result structure
        assert "phase_states" in result
        assert "progress" in result
        assert "current_phase" in result

        # Verify phases were detected
        assert "Reading" in result["phase_states"]
        assert "Iteration 1" in result["phase_states"]
