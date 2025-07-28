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
        """创建测试用的QueueManager，每个测试使用新实例"""
        # 重置单例以确保每个测试独立
        QueueManager._instance = None
        
        # Mock TaskDatabase to avoid file system dependencies
        with patch('autogluon.assistant.webui.backend.queue.manager.TaskDatabase') as mock_db_class:
            mock_db = Mock(spec=TaskDatabase)
            mock_db_class.return_value = mock_db
            
            manager = QueueManager()
            manager.db = mock_db
            
            # 确保停止任何运行的线程
            yield manager
            
            # 清理
            manager.stop()
            # 等待线程结束
            if manager._executor_thread and manager._executor_thread.is_alive():
                manager._executor_thread.join(timeout=1)
    
    def test_singleton_pattern(self):
        """测试单例模式"""
        # 重置单例
        QueueManager._instance = None
        
        manager1 = QueueManager()
        manager2 = QueueManager()
        
        assert manager1 is manager2
        
        # 清理
        manager1.stop()
    
    def test_submit_task(self, queue_manager):
        """测试任务提交"""
        # Mock数据库返回
        queue_manager.db.add_task.return_value = 0
        
        # 提交任务
        position = queue_manager.submit_task(
            "test_task_123",
            {"cmd": ["echo", "test"], "max_iter": 5},
            {"AWS_ACCESS_KEY_ID": "test_key"}
        )
        
        # 验证
        assert position == 0
        queue_manager.db.add_task.assert_called_once_with(
            "test_task_123",
            {"cmd": ["echo", "test"], "max_iter": 5},
            {"AWS_ACCESS_KEY_ID": "test_key"}
        )
    
    def test_cancel_task(self, queue_manager):
        """测试取消任务"""
        # Mock返回值
        queue_manager.db.cancel_task.return_value = True
        
        # 取消任务
        success = queue_manager.cancel_task("task_to_cancel")
        
        # 验证
        assert success is True
        queue_manager.db.cancel_task.assert_called_once_with("task_to_cancel")
    
    def test_get_task_status(self, queue_manager):
        """测试获取任务状态"""
        # Mock返回值
        expected_status = {
            "task_id": "test_task",
            "status": "running",
            "position": 0
        }
        queue_manager.db.get_task_status.return_value = expected_status
        
        # 获取状态
        status = queue_manager.get_task_status("test_task")
        
        # 验证
        assert status == expected_status
        queue_manager.db.get_task_status.assert_called_once_with("test_task")
    
    def test_complete_task_by_run_id(self, queue_manager):
        """测试通过run_id完成任务"""
        # Mock返回值
        queue_manager.db.get_task_by_run_id.return_value = "task_123"
        
        # 完成任务
        queue_manager.complete_task_by_run_id("run_456")
        
        # 验证调用顺序
        queue_manager.db.get_task_by_run_id.assert_called_once_with("run_456")
        queue_manager.db.complete_task.assert_called_once_with("task_123")
    
    @patch('autogluon.assistant.webui.backend.utils.start_run')
    def test_executor_loop_single_task(self, mock_start_run, queue_manager):
        """测试执行器循环处理单个任务"""
        # Mock start_run 返回 run_id
        mock_start_run.return_value = "run_123"
        
        # 设置任务队列行为
        queue_manager.db.get_next_task.side_effect = [
            ("task1", {"cmd": ["echo", "test"]}, {"AWS_KEY": "value"}),  # 第一次返回任务
            None,  # 第二次返回None
            None   # 第三次返回None（确保循环结束）
        ]
        
        # 启动执行器
        queue_manager.start()
        
        # 等待任务被处理
        time.sleep(2)
        
        # 停止执行器
        queue_manager.stop()
        
        # 验证
        mock_start_run.assert_called_once_with(
            "task1", 
            ["echo", "test"],
            {"AWS_KEY": "value"}
        )
        queue_manager.db.update_task_run_id.assert_called_once_with("task1", "run_123")
        queue_manager.db.cleanup_stale_tasks.assert_called()
    
    @patch('autogluon.assistant.webui.backend.utils.start_run')
    def test_executor_loop_multiple_tasks(self, mock_start_run, queue_manager):
        """测试执行器循环处理多个任务"""
        # Mock start_run 返回不同的 run_id
        mock_start_run.side_effect = ["run_001", "run_002"]
        
        # 设置任务队列行为
        queue_manager.db.get_next_task.side_effect = [
            ("task1", {"cmd": ["cmd1"]}, None),
            ("task2", {"cmd": ["cmd2"]}, None),
            None  # 结束循环
        ]
        
        # 启动执行器
        queue_manager.start()
        
        # 等待任务被处理
        time.sleep(3)
        
        # 停止执行器
        queue_manager.stop()
        
        # 验证两个任务都被处理
        assert mock_start_run.call_count == 2
        assert queue_manager.db.update_task_run_id.call_count == 2
    
    @patch('autogluon.assistant.webui.backend.utils.start_run')
    def test_executor_handles_start_run_failure(self, mock_start_run, queue_manager):
        """测试执行器处理start_run失败的情况"""
        # Mock start_run 抛出异常
        mock_start_run.side_effect = Exception("Failed to start")
        
        # 设置任务队列行为
        queue_manager.db.get_next_task.side_effect = [
            ("task_fail", {"cmd": ["bad_cmd"]}, None),
            None
        ]
        
        # 启动执行器
        queue_manager.start()
        
        # 等待处理
        time.sleep(2)
        
        # 停止执行器
        queue_manager.stop()
        
        # 验证失败的任务被标记为完成（从队列中移除）
        queue_manager.db.complete_task.assert_called_once_with("task_fail")
    
    def test_start_stop_executor(self, queue_manager):
        """测试启动和停止执行器线程"""
        # Mock get_next_task 始终返回 None
        queue_manager.db.get_next_task.return_value = None
        
        # 验证初始状态
        assert queue_manager._executor_thread is None or not queue_manager._executor_thread.is_alive()
        
        # 启动
        queue_manager.start()
        time.sleep(0.5)
        
        # 验证线程在运行
        assert queue_manager._executor_thread is not None
        assert queue_manager._executor_thread.is_alive()
        
        # 停止
        queue_manager.stop()
        
        # 验证线程已停止
        assert queue_manager._stop_event.is_set()
        # 等待线程结束
        queue_manager._executor_thread.join(timeout=2)
        assert not queue_manager._executor_thread.is_alive()
    
    def test_double_start(self, queue_manager):
        """测试重复启动不会创建多个线程"""
        # Mock get_next_task
        queue_manager.db.get_next_task.return_value = None
        
        # 第一次启动
        queue_manager.start()
        first_thread = queue_manager._executor_thread
        
        # 第二次启动
        queue_manager.start()
        second_thread = queue_manager._executor_thread
        
        # 验证是同一个线程
        assert first_thread is second_thread
        
        # 清理
        queue_manager.stop()
    
    def test_cleanup_stale_tasks_called(self, queue_manager):
        """测试清理过期任务被定期调用"""
        # Mock get_next_task 返回 None
        queue_manager.db.get_next_task.return_value = None
        
        # 启动执行器
        queue_manager.start()
        
        # 等待几个循环
        time.sleep(3.5)
        
        # 停止执行器
        queue_manager.stop()
        
        # 验证 cleanup_stale_tasks 被调用多次
        assert queue_manager.db.cleanup_stale_tasks.call_count >= 3
    
    def test_get_queue_info(self, queue_manager):
        """测试获取队列信息"""
        # Mock返回值
        expected_info = {
            "queued": 5,
            "running": 1,
            "total_waiting": 6
        }
        queue_manager.db.get_queue_info.return_value = expected_info
        
        # 获取队列信息
        info = queue_manager.get_queue_info()
        
        # 验证
        assert info == expected_info
        queue_manager.db.get_queue_info.assert_called_once()