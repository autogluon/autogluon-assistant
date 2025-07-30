# tests/unittests/webui/test_backend/test_e2e_queue_flow.py

import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from autogluon.assistant.webui.backend.app import create_app
from autogluon.assistant.webui.backend.queue import get_queue_manager
from autogluon.assistant.webui.backend.queue.models import TaskDatabase


class TestE2EQueueFlow:

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_queue.db"
            yield str(db_path)

    @pytest.fixture
    def app(self, temp_db):
        """Create test application with temporary database"""
        # Patch TaskDatabase to use our temporary database
        with patch("autogluon.assistant.webui.backend.queue.manager.TaskDatabase") as mock_db_class:
            # Create a real TaskDatabase instance with temp path
            test_db = TaskDatabase(temp_db)
            mock_db_class.return_value = test_db

            # Reset the QueueManager singleton
            from autogluon.assistant.webui.backend.queue.manager import QueueManager

            QueueManager._instance = None

            # Create app
            app = create_app()
            app.config["TESTING"] = True

            yield app

            # Cleanup: stop queue manager
            queue_manager = get_queue_manager()
            queue_manager.stop()

    @pytest.fixture
    def client(self, app):
        """Create test client"""
        return app.test_client()

    @patch("subprocess.Popen")
    def test_complete_task_flow(self, mock_popen, client):
        """Test complete flow from submission to completion"""
        # Mock subprocess
        mock_process = Mock()
        mock_process.stdout = iter(["BRIEF Starting task\n", "INFO Task completed\n"])
        mock_process.poll.return_value = 0
        mock_process.wait.return_value = 0
        mock_process.returncode = 0
        mock_process.stdin = Mock()
        mock_process.stdin.write = Mock()
        mock_process.stdin.flush = Mock()
        mock_popen.return_value = mock_process

        # 1. Submit task
        response = client.post(
            "/api/run", json={"data_src": "/tmp/test", "max_iter": 1, "verbosity": "1", "config_path": "test.yaml"}
        )

        assert response.status_code == 200
        data = response.json
        task_id = data["task_id"]
        assert data["position"] == 0  # Should be 0 since database is empty

        # 2. Check queue status
        response = client.get(f"/api/queue/status/{task_id}")
        status = response.json
        assert status["status"] in ["queued", "running"]
        assert status["task_id"] == task_id

        # 3. Wait for task to start (queue manager picks it up)
        time.sleep(2)

        # 4. Check status again - should have run_id now
        response = client.get(f"/api/queue/status/{task_id}")
        status = response.json

        # If task has started, it should have a run_id
        if status["status"] == "running":
            assert status["run_id"] is not None
            run_id = status["run_id"]

            # 5. Get logs using run_id
            response = client.get("/api/logs", query_string={"run_id": run_id})
            logs = response.json["lines"]
            assert len(logs) > 0

            # 6. Check task completion status
            response = client.get("/api/status", query_string={"run_id": run_id})
            assert response.json["finished"] is True

        # 7. Verify queue is empty after completion
        response = client.get("/api/queue/info")
        queue_info = response.json
        assert queue_info["queued"] == 0
        assert queue_info["running"] == 0

    @patch("subprocess.Popen")
    def test_queue_position_with_multiple_tasks(self, mock_popen, client):
        """Test queue positions when multiple tasks are submitted"""
        # Mock subprocess that runs slowly
        mock_process = Mock()
        mock_process.stdout = iter(
            [
                "BRIEF Task running slowly\n",
            ]
        )
        mock_process.poll.return_value = None  # Still running
        mock_process.wait.return_value = None
        mock_process.stdin = Mock()
        mock_popen.return_value = mock_process

        # Submit first task
        response1 = client.post(
            "/api/run", json={"data_src": "/tmp/test1", "max_iter": 1, "verbosity": "1", "config_path": "test.yaml"}
        )
        assert response1.status_code == 200
        data1 = response1.json
        assert data1["position"] == 0

        # Submit second task
        response2 = client.post(
            "/api/run", json={"data_src": "/tmp/test2", "max_iter": 1, "verbosity": "1", "config_path": "test.yaml"}
        )
        assert response2.status_code == 200
        data2 = response2.json
        assert data2["position"] == 1  # Should be queued behind first task

        # Submit third task
        response3 = client.post(
            "/api/run", json={"data_src": "/tmp/test3", "max_iter": 1, "verbosity": "1", "config_path": "test.yaml"}
        )
        assert response3.status_code == 200
        data3 = response3.json
        assert data3["position"] == 2  # Should be queued behind first two tasks

    @patch("subprocess.Popen")
    def test_cancel_queued_task(self, mock_popen, client):
        """Test cancelling a queued task"""
        # Mock subprocess
        mock_process = Mock()
        mock_process.stdout = iter([])
        mock_process.poll.return_value = None
        mock_process.stdin = Mock()
        mock_popen.return_value = mock_process

        # Submit two tasks
        response1 = client.post(
            "/api/run", json={"data_src": "/tmp/test1", "max_iter": 1, "verbosity": "1", "config_path": "test.yaml"}
        )
        task1_id = response1.json["task_id"]

        response2 = client.post(
            "/api/run", json={"data_src": "/tmp/test2", "max_iter": 1, "verbosity": "1", "config_path": "test.yaml"}
        )
        task2_id = response2.json["task_id"]

        # Cancel the second (queued) task
        response = client.post("/api/cancel", json={"task_id": task2_id})
        assert response.status_code == 200
        assert response.json["cancelled"] is True

        # Verify task2 is gone
        response = client.get(f"/api/queue/status/{task2_id}")
        assert response.status_code == 404  # Task not found

        # Verify task1 is still there
        response = client.get(f"/api/queue/status/{task1_id}")
        assert response.status_code == 200
