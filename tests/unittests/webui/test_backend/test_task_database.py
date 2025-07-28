# tests/unittests/webui/test_backend/test_task_database.py

import pytest
import tempfile
import subprocess
import os
from pathlib import Path

from autogluon.assistant.webui.backend.queue.models import TaskDatabase


class TestTaskDatabase:
    
    @pytest.fixture
    def db(self):
        """使用queuedb脚本创建测试数据库"""
        # 创建临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            # 数据库路径
            db_path = Path(temp_dir) / "test_queue.db"
            
            # 使用 queuedb 脚本创建数据库，指定自定义路径
            script_path = "src/autogluon/assistant/webui/backend/queue/queuedb"
            result = subprocess.run([script_path, "--db-path", str(db_path), "create"], 
                                  capture_output=True, text=True)
            
            # 验证脚本执行成功
            assert result.returncode == 0, f"Failed to create db: {result.stderr}"
            assert db_path.exists(), "Database file was not created"
            
            # 创建 TaskDatabase 实例
            db = TaskDatabase(str(db_path))
            
            yield db
    
    def test_queuedb_script_create(self):
        """测试 queuedb create 命令"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_create.db"
            script_path = "src/autogluon/assistant/webui/backend/queue/queuedb"
            
            # 第一次创建
            result = subprocess.run([script_path, "--db-path", str(db_path), "create"], 
                                  capture_output=True, text=True)
            assert result.returncode == 0
            assert "Database created and initialized" in result.stdout
            assert db_path.exists()
            
            # 第二次创建（应该提示已存在）
            result = subprocess.run([script_path, "--db-path", str(db_path), "create"], 
                                  capture_output=True, text=True)
            assert result.returncode == 0
            assert "already exist" in result.stdout
    
    def test_queuedb_script_default_path(self):
        """测试 queuedb 使用默认路径"""
        # 设置临时 HOME
        with tempfile.TemporaryDirectory() as temp_home:
            old_home = os.environ.get('HOME')
            os.environ['HOME'] = temp_home
            
            try:
                script_path = "src/autogluon/assistant/webui/backend/queue/queuedb"
                
                # 不指定 --db-path，应该使用默认路径
                result = subprocess.run([script_path, "create"], 
                                      capture_output=True, text=True)
                assert result.returncode == 0
                
                # 验证默认路径的数据库被创建
                default_db = Path(temp_home) / '.autogluon_assistant' / 'webui_queue.db'
                assert default_db.exists()
                
            finally:
                if old_home is not None:
                    os.environ['HOME'] = old_home
                else:
                    del os.environ['HOME']
    
    def test_queuedb_script_help(self):
        """测试 queuedb --help 命令"""
        script_path = "src/autogluon/assistant/webui/backend/queue/queuedb"
        
        result = subprocess.run([script_path, "--help"], 
                              capture_output=True, text=True)
        assert result.returncode == 0
        assert "Usage:" in result.stdout
        assert "--db-path" in result.stdout
        assert "create" in result.stdout
        assert "reset" in result.stdout
        assert "dump" in result.stdout
    
    def test_add_task_queue_position(self, db):
        """测试任务添加和队列位置计算"""
        # 第一个任务，位置应该是0
        pos1 = db.add_task("task1", {"cmd": ["echo", "hello"]})
        assert pos1 == 0
        
        # 第二个任务，位置应该是1
        pos2 = db.add_task("task2", {"cmd": ["echo", "world"]})
        assert pos2 == 1
        
        # 获取第一个任务并标记为运行中
        task = db.get_next_task()
        assert task[0] == "task1"
        
        # 第三个任务，位置应该是2（因为有1个running + 1个queued）
        pos3 = db.add_task("task3", {"cmd": ["echo", "test"]})
        assert pos3 == 2
        
        # 完成 task1
        db.complete_task("task1")
        
        # 第四个任务，位置应该是2（因为有2个queued）
        pos4 = db.add_task("task4", {"cmd": ["echo", "four"]})
        assert pos4 == 2
        
        # 获取下一个任务（task2 变为 running）
        task = db.get_next_task()
        assert task[0] == "task2"
        
        # 第五个任务，位置应该是3（因为有1个running + 2个queued）
        pos5 = db.add_task("task5", {"cmd": ["echo", "five"]})
        assert pos5 == 3
    
    def test_get_next_task_fifo(self, db):
        """测试FIFO顺序获取任务"""
        # 添加多个任务
        db.add_task("task1", {"cmd": ["cmd1"]})
        db.add_task("task2", {"cmd": ["cmd2"]})
        db.add_task("task3", {"cmd": ["cmd3"]})
        
        # 按顺序获取
        task1 = db.get_next_task()
        assert task1[0] == "task1"
        
        # 完成task1
        db.complete_task("task1")
        
        # 获取下一个
        task2 = db.get_next_task()
        assert task2[0] == "task2"
    
    def test_concurrent_task_limit(self, db):
        """测试同时只能运行一个任务"""
        db.add_task("task1", {"cmd": ["sleep", "10"]})
        db.add_task("task2", {"cmd": ["echo", "hi"]})
        
        # 获取第一个任务
        task1 = db.get_next_task()
        assert task1 is not None
        
        # 尝试获取第二个任务（应该返回None）
        task2 = db.get_next_task()
        assert task2 is None
    
    def test_update_run_id(self, db):
        """测试更新run_id"""
        db.add_task("task123", {"cmd": ["mlzero"]})
        db.get_next_task()  # 标记为运行中
        
        # 更新run_id
        db.update_task_run_id("task123", "run456")
        
        # 验证
        status = db.get_task_status("task123")
        assert status["run_id"] == "run456"
    
    def test_cancel_queued_task(self, db):
        """测试取消排队中的任务"""
        db.add_task("task1", {"cmd": ["cmd1"]})
        db.add_task("task2", {"cmd": ["cmd2"]})
        
        # 取消排队中的task2
        success = db.cancel_task("task2")
        assert success is True
        
        # 验证task2不存在
        status = db.get_task_status("task2")
        assert status is None
    
    def test_cannot_cancel_running_task(self, db):
        """测试不能通过cancel_task取消运行中的任务"""
        db.add_task("task1", {"cmd": ["cmd1"]})
        db.get_next_task()  # 标记为运行中
        
        # 尝试取消运行中的任务
        success = db.cancel_task("task1")
        assert success is False
    
    def test_queuedb_script_dump(self, db):
        """测试 queuedb dump 命令"""
        # 添加一些任务
        db.add_task("test_task_1", {"cmd": ["echo", "test1"]})
        db.add_task("test_task_2", {"cmd": ["echo", "test2"]})
        
        # 获取一个任务（使其变为running状态）
        db.get_next_task()
        
        # 使用相同的数据库路径
        db_path = db.db_path
        
        script_path = "src/autogluon/assistant/webui/backend/queue/queuedb"
        result = subprocess.run([script_path, "--db-path", db_path, "dump"], 
                              capture_output=True, text=True)
        
        assert result.returncode == 0
        assert "Database:" in result.stdout
        assert db_path in result.stdout
        assert "Task Queue Summary" in result.stdout
        assert "Total tasks: 2" in result.stdout
        assert "Queued: 1" in result.stdout
        assert "Running: 1" in result.stdout
    
    def test_queuedb_script_reset(self, db):
        """测试 queuedb reset 命令"""
        # 添加一些任务
        db.add_task("task_to_reset_1", {"cmd": ["echo", "reset1"]})
        db.add_task("task_to_reset_2", {"cmd": ["echo", "reset2"]})
        
        # 验证任务存在
        assert db.get_task_status("task_to_reset_1") is not None
        
        # 使用相同的数据库路径
        db_path = db.db_path
        
        script_path = "src/autogluon/assistant/webui/backend/queue/queuedb"
        result = subprocess.run([script_path, "--db-path", db_path, "reset"], 
                              capture_output=True, text=True)
        
        assert result.returncode == 0
        assert "All tasks cleared" in result.stdout
        
        # 验证任务被清除
        info = db.get_queue_info()
        assert info['queued'] == 0
        assert info['running'] == 0
    
    def test_cleanup_stale_tasks(self, db):
        """测试清理超时任务"""
        from datetime import datetime, timedelta
        
        # 添加任务
        db.add_task("stale_task", {"cmd": ["hang"]})
        
        # 直接修改数据库，设置为2小时前开始运行
        import sqlite3
        conn = sqlite3.connect(db.db_path)
        try:
            past_time = (datetime.now() - timedelta(hours=2)).isoformat()
            conn.execute(
                "UPDATE tasks SET status='running', started_at=? WHERE task_id=?",
                (past_time, "stale_task")
            )
            conn.commit()
        finally:
            conn.close()
        
        # 清理超时任务（超时1小时）
        db.cleanup_stale_tasks(timeout_seconds=3600)
        
        # 验证任务被删除
        status = db.get_task_status("stale_task")
        assert status is None
