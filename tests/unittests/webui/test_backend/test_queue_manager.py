# tests/unittests/webui/test_backend/test_queue_manager.py

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock

from autogluon.assistant.webui.backend.queue.manager import QueueManager
from autogluon.assistant.webui.backend.queue.models import TaskDatabase


class TestQueueManager:
    
    @pytest.fixture
    def queue_manager(self):
        """Create test QueueManager, ensure each test uses a new instance"""
        # Reset singleton to ensure test isolation
        QueueManager._instance = None
        
        # Mock TaskDatabase to avoid file system dependencies
        with patch('autogluon.assistant.webui.backend.queue.manager.TaskDatabase') as mock_db_class:
            mock_db = Mock(spec=TaskDatabase)
            mock_db_class.return_value = mock_db
            
            manager = QueueManager()
            manager.db = mock_db
            
            # Ensure any running threads are stopped
            yield manager
            
            # Cleanup
            manager.stop()
            # Wait for thread to end
            if manager._executor_thread and manager._executor_thread.is_alive():
                manager._executor_thread.join(timeout=1)
    
    def test_singleton_pattern(self):
        """Test singleton pattern"""
        # Reset singleton
        QueueManager._instance = None
        
        manager1 = QueueManager()
        manager2 = QueueManager()
        
        assert manager1 is manager2
        
        # Cleanup
        manager1.stop()
    
    def test_submit_task(self, queue_manager):
        """Test task submission"""
        # Mock database return
        queue_manager.db.add_task.return_value = 0
        
        # Submit task
        position = queue_manager.submit_task(
            "test_task_123",
            {"cmd": ["echo", "test"], "max_iter": 5},
            {"AWS_ACCESS_KEY_ID": "test_key"}
        )
        
        # Verify
        assert position == 0
        queue_manager.db.add_task.assert_called_once_with(
            "test_task_123",
            {"cmd": ["echo", "test"], "max_iter": 5},
            {"AWS_ACCESS_KEY_ID": "test_key"}
        )
    
    def test_cancel_task(self, queue_manager):
        """Test cancel task"""
        # Mock return value
        queue_manager.db.cancel_task.return_value = True
        
        # Cancel task
        success = queue_manager.cancel_task("task_to_cancel")
        
        # Verify
        assert success is True
        queue_manager.db.cancel_task.assert_called_once_with("task_to_cancel")
    
    def test_get_task_status(self, queue_manager):
        """Test get task status"""
        # Mock return value
        expected_status = {
            "task_id": "test_task",
            "status": "running",
            "position": 0
        }
        queue_manager.db.get_task_status.return_value = expected_status
        
        # Get status
        status = queue_manager.get_task_status("test_task")
        
        # Verify
        assert status == expected_status
        queue_manager.db.get_task_status.assert_called_once_with("test_task")
    
    def test_complete_task_by_run_id(self, queue_manager):
        """Test complete task by run_id"""
        # Mock return value
        queue_manager.db.get_task_by_run_id.return_value = "task_123"
        
        # Complete task
        queue_manager.complete_task_by_run_id("run_456")
        
        # Verify call sequence
        queue_manager.db.get_task_by_run_id.assert_called_once_with("run_456")
        queue_manager.db.complete_task.assert_called_once_with("task_123")
    
    @patch('autogluon.assistant.webui.backend.queue.manager.start_run')
    def test_executor_loop_single_task(self, mock_start_run, queue_manager):
        """Test executor loop processing single task"""
        # Mock start_run returns run_id
        mock_start_run.return_value = "run_123"
        
        # Setup task queue behavior
        queue_manager.db.get_next_task.side_effect = [
            ("task1", {"cmd": ["echo", "test"]}, {"AWS_KEY": "value"}),  # First return task
            None,  # Second return None
            None   # Third return None (ensure loop ends)
        ]
        
        # Start executor
        queue_manager.start()
        
        # Wait for task to be processed
        time.sleep(2)
        
        # Stop executor
        queue_manager.stop()
        
        # Verify
        mock_start_run.assert_called_once_with(
            "task1", 
            ["echo", "test"],
            {"AWS_KEY": "value"}
        )
        queue_manager.db.update_task_run_id.assert_called_once_with("task1", "run_123")
        queue_manager.db.cleanup_stale_tasks.assert_called()
    
    @patch('autogluon.assistant.webui.backend.queue.manager.start_run')
    def test_executor_loop_multiple_tasks(self, mock_start_run, queue_manager):
        """Test executor loop processing multiple tasks"""
        # Mock start_run returns different run_ids
        mock_start_run.side_effect = ["run_001", "run_002"]
        
        # Setup task queue behavior
        queue_manager.db.get_next_task.side_effect = [
            ("task1", {"cmd": ["cmd1"]}, None),
            ("task2", {"cmd": ["cmd2"]}, None),
            None  # End loop
        ]
        
        # Start executor
        queue_manager.start()
        
        # Wait for tasks to be processed
        time.sleep(3)
        
        # Stop executor
        queue_manager.stop()
        
        # Verify both tasks were processed
        assert mock_start_run.call_count == 2
        assert queue_manager.db.update_task_run_id.call_count == 2
    
    @patch('autogluon.assistant.webui.backend.queue.manager.start_run')
    def test_executor_handles_start_run_failure(self, mock_start_run, queue_manager):
        """Test executor handles start_run failure"""
        # Mock start_run throws exception
        mock_start_run.side_effect = Exception("Failed to start")
        
        # Setup task queue behavior
        queue_manager.db.get_next_task.side_effect = [
            ("task_fail", {"cmd": ["bad_cmd"]}, None),
            None
        ]
        
        # Start executor
        queue_manager.start()
        
        # Wait for processing
        time.sleep(2)
        
        # Stop executor
        queue_manager.stop()
        
        # Verify failed task is marked as complete (removed from queue)
        queue_manager.db.complete_task.assert_called_once_with("task_fail")
    
    def test_start_stop_executor(self, queue_manager):
        """Test start and stop executor thread"""
        # Mock get_next_task always returns None
        queue_manager.db.get_next_task.return_value = None
        
        # Verify initial state
        assert queue_manager._executor_thread is None or not queue_manager._executor_thread.is_alive()
        
        # Start
        queue_manager.start()
        time.sleep(0.5)
        
        # Verify thread is running
        assert queue_manager._executor_thread is not None
        assert queue_manager._executor_thread.is_alive()
        
        # Stop
        queue_manager.stop()
        
        # Verify thread has stopped
        assert queue_manager._stop_event.is_set()
        # Wait for thread to end
        queue_manager._executor_thread.join(timeout=2)
        assert not queue_manager._executor_thread.is_alive()
    
    def test_double_start(self, queue_manager):
        """Test repeated start does not create multiple threads"""
        # Mock get_next_task
        queue_manager.db.get_next_task.return_value = None
        
        # First start
        queue_manager.start()
        first_thread = queue_manager._executor_thread
        
        # Second start
        queue_manager.start()
        second_thread = queue_manager._executor_thread
        
        # Verify it's the same thread
        assert first_thread is second_thread
        
        # Cleanup
        queue_manager.stop()
    
    def test_cleanup_stale_tasks_called(self, queue_manager):
        """Test cleanup of expired tasks is called periodically"""
        # Mock get_next_task returns None
        queue_manager.db.get_next_task.return_value = None
        
        # Start executor
        queue_manager.start()
        
        # Wait for several loops
        time.sleep(3.5)
        
        # Stop executor
        queue_manager.stop()
        
        # Verify cleanup_stale_tasks was called multiple times
        assert queue_manager.db.cleanup_stale_tasks.call_count >= 3
    
    def test_get_queue_info(self, queue_manager):
        """Test get queue information"""
        # Mock return value
        expected_info = {
            "queued": 5,
            "running": 1,
            "total_waiting": 6
        }
        queue_manager.db.get_queue_info.return_value = expected_info
        
        # Get queue info
        info = queue_manager.get_queue_info()
        
        # Verify
        assert info == expected_info
        queue_manager.db.get_queue_info.assert_called_once()
