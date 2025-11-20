import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from autogluon.assistant.webui.backend.queue.models import TaskDatabase


class TestTaskDatabase:

    @pytest.fixture
    def db(self):
        """Create a test database using the queuedb script"""
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Define the database path
            db_path = Path(temp_dir) / "test_queue.db"

            # Get the absolute path to the queuedb script
            script_dir = (
                Path(__file__).parent.parent.parent.parent.parent
                / "src"
                / "autogluon"
                / "assistant"
                / "webui"
                / "backend"
                / "queue"
            )
            script_path = script_dir / "queuedb"

            # Create database using the queuedb script with a custom path
            result = subprocess.run(
                [str(script_path), "--db-path", str(db_path), "create"], capture_output=True, text=True
            )

            # Verify that the script ran successfully
            assert result.returncode == 0, f"Failed to create db: {result.stderr}"
            assert db_path.exists(), "Database file was not created"

            # Create TaskDatabase instance
            db = TaskDatabase(str(db_path))

            yield db

    def test_queuedb_script_create(self):
        """Test the queuedb create command"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_create.db"

            # Get the absolute path to the queuedb script
            script_dir = (
                Path(__file__).parent.parent.parent.parent.parent
                / "src"
                / "autogluon"
                / "assistant"
                / "webui"
                / "backend"
                / "queue"
            )
            script_path = script_dir / "queuedb"

            # First creation
            result = subprocess.run(
                [str(script_path), "--db-path", str(db_path), "create"], capture_output=True, text=True
            )
            assert result.returncode == 0
            assert "Database created and initialized" in result.stdout
            assert db_path.exists()

            # Second creation (should show already exists)
            result = subprocess.run(
                [str(script_path), "--db-path", str(db_path), "create"], capture_output=True, text=True
            )
            assert result.returncode == 0
            assert "already exist" in result.stdout

    def test_queuedb_script_default_path(self):
        """Test queuedb with default path"""
        # Set temporary HOME
        with tempfile.TemporaryDirectory() as temp_home:
            old_home = os.environ.get("HOME")
            os.environ["HOME"] = temp_home

            try:
                # Get the absolute path to the queuedb script
                script_dir = (
                    Path(__file__).parent.parent.parent.parent.parent
                    / "src"
                    / "autogluon"
                    / "assistant"
                    / "webui"
                    / "backend"
                    / "queue"
                )
                script_path = script_dir / "queuedb"

                # Do not specify --db-path, should use default path
                result = subprocess.run([str(script_path), "create"], capture_output=True, text=True)
                assert result.returncode == 0

                # Check if default path db was created
                default_db = Path(temp_home) / ".autogluon_assistant" / "webui_queue.db"
                assert default_db.exists()

            finally:
                if old_home is not None:
                    os.environ["HOME"] = old_home
                else:
                    del os.environ["HOME"]

    def test_queuedb_script_help(self):
        """Test queuedb --help command"""
        # Get the absolute path to the queuedb script
        script_dir = (
            Path(__file__).parent.parent.parent.parent.parent
            / "src"
            / "autogluon"
            / "assistant"
            / "webui"
            / "backend"
            / "queue"
        )
        script_path = script_dir / "queuedb"

        result = subprocess.run([str(script_path), "--help"], capture_output=True, text=True)
        assert result.returncode == 0
        assert "Usage:" in result.stdout
        assert "--db-path" in result.stdout
        assert "create" in result.stdout
        assert "reset" in result.stdout
        assert "dump" in result.stdout

    def test_add_task_queue_position(self, db):
        """Test task addition and queue position calculation"""
        # First task, position should be 0
        pos1 = db.add_task("task1", {"cmd": ["echo", "hello"]})
        assert pos1 == 0

        # Second task, position should be 1
        pos2 = db.add_task("task2", {"cmd": ["echo", "world"]})
        assert pos2 == 1

        # Get the first task and mark it as running
        task = db.get_next_task()
        assert task[0] == "task1"

        # Third task, position should be 2 (1 running + 1 queued)
        pos3 = db.add_task("task3", {"cmd": ["echo", "test"]})
        assert pos3 == 2

        # Complete task1
        db.complete_task("task1")

        # Fourth task, position should be 2 (2 queued)
        pos4 = db.add_task("task4", {"cmd": ["echo", "four"]})
        assert pos4 == 2

        # Get next task (task2 becomes running)
        task = db.get_next_task()
        assert task[0] == "task2"

        # Fifth task, position should be 3 (1 running + 2 queued)
        pos5 = db.add_task("task5", {"cmd": ["echo", "five"]})
        assert pos5 == 3

    def test_get_next_task_fifo(self, db):
        """Test FIFO order of getting tasks"""
        # Add multiple tasks
        db.add_task("task1", {"cmd": ["cmd1"]})
        db.add_task("task2", {"cmd": ["cmd2"]})
        db.add_task("task3", {"cmd": ["cmd3"]})

        # Get in order
        task1 = db.get_next_task()
        assert task1[0] == "task1"

        # Complete task1
        db.complete_task("task1")

        # Get next
        task2 = db.get_next_task()
        assert task2[0] == "task2"

    def test_concurrent_task_limit(self, db):
        """Test that only one task can run concurrently"""
        db.add_task("task1", {"cmd": ["sleep", "10"]})
        db.add_task("task2", {"cmd": ["echo", "hi"]})

        # Get the first task
        task1 = db.get_next_task()
        assert task1 is not None

        # Try to get the second task (should return None)
        task2 = db.get_next_task()
        assert task2 is None

    def test_update_run_id(self, db):
        """Test updating run_id"""
        db.add_task("task123", {"cmd": ["mlzero"]})
        db.get_next_task()  # Mark as running

        # Update run_id
        db.update_task_run_id("task123", "run456")

        # Verify
        status = db.get_task_status("task123")
        assert status["run_id"] == "run456"

    def test_cancel_queued_task(self, db):
        """Test cancelling a queued task"""
        db.add_task("task1", {"cmd": ["cmd1"]})
        db.add_task("task2", {"cmd": ["cmd2"]})

        # Cancel task2 (queued)
        success = db.cancel_task("task2")
        assert success is True

        # Verify task2 no longer exists
        status = db.get_task_status("task2")
        assert status is None

    def test_cannot_cancel_running_task(self, db):
        """Test that a running task cannot be cancelled using cancel_task"""
        db.add_task("task1", {"cmd": ["cmd1"]})
        db.get_next_task()  # Mark as running

        # Attempt to cancel running task
        success = db.cancel_task("task1")
        assert success is False

    def test_queuedb_script_dump(self, db):
        """Test queuedb dump command"""
        # Add some tasks
        db.add_task("test_task_1", {"cmd": ["echo", "test1"]})
        db.add_task("test_task_2", {"cmd": ["echo", "test2"]})

        # Get a task to mark it as running
        db.get_next_task()

        # Use same database path
        db_path = db.db_path

        # Get the absolute path to the queuedb script
        script_dir = (
            Path(__file__).parent.parent.parent.parent.parent
            / "src"
            / "autogluon"
            / "assistant"
            / "webui"
            / "backend"
            / "queue"
        )
        script_path = script_dir / "queuedb"

        result = subprocess.run([str(script_path), "--db-path", db_path, "dump"], capture_output=True, text=True)

        assert result.returncode == 0
        assert "Database:" in result.stdout
        assert db_path in result.stdout
        assert "Task Queue Summary" in result.stdout
        assert "Total tasks: 2" in result.stdout
        assert "Queued: 1" in result.stdout
        assert "Running: 1" in result.stdout

    def test_queuedb_script_reset(self, db):
        """Test queuedb reset command"""
        # Add some tasks
        db.add_task("task_to_reset_1", {"cmd": ["echo", "reset1"]})
        db.add_task("task_to_reset_2", {"cmd": ["echo", "reset2"]})

        # Ensure task exists
        assert db.get_task_status("task_to_reset_1") is not None

        # Use same database path
        db_path = db.db_path

        # Get the absolute path to the queuedb script
        script_dir = (
            Path(__file__).parent.parent.parent.parent.parent
            / "src"
            / "autogluon"
            / "assistant"
            / "webui"
            / "backend"
            / "queue"
        )
        script_path = script_dir / "queuedb"

        result = subprocess.run([str(script_path), "--db-path", db_path, "reset"], capture_output=True, text=True)

        assert result.returncode == 0
        assert "All tasks cleared" in result.stdout

        # Verify tasks are cleared
        info = db.get_queue_info()
        assert info["queued"] == 0
        assert info["running"] == 0

    def test_cleanup_stale_tasks(self, db):
        """Test cleanup of stale (timeout) tasks"""
        from datetime import datetime, timedelta

        # Add a task
        db.add_task("stale_task", {"cmd": ["hang"]})

        # Modify DB to simulate a task that started 2 hours ago
        import sqlite3

        conn = sqlite3.connect(db.db_path)
        try:
            past_time = (datetime.now() - timedelta(hours=2)).isoformat()
            conn.execute("UPDATE tasks SET status='running', started_at=? WHERE task_id=?", (past_time, "stale_task"))
            conn.commit()
        finally:
            conn.close()

        # Cleanup stale tasks with 1-hour timeout
        db.cleanup_stale_tasks(timeout_seconds=3600)

        # Verify the task was removed
        status = db.get_task_status("stale_task")
        assert status is None
