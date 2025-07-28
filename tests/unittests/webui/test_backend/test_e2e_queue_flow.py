# tests/unittests/webui/test_backend/test_e2e_queue_flow.py

import pytest
import requests
import time
from unittest.mock import patch, Mock

from autogluon.assistant.webui.backend.app import create_app


class TestE2EQueueFlow:
    
    @pytest.fixture
    def app(self):
        """创建测试应用"""
        app = create_app()
        app.config['TESTING'] = True
        return app
    
    @pytest.fixture
    def client(self, app):
        """创建测试客户端"""
        return app.test_client()
    
    @patch('subprocess.Popen')
    def test_complete_task_flow(self, mock_popen, client):
        """测试从提交到完成的完整流程"""
        # 模拟子进程
        mock_process = Mock()
        mock_process.stdout = iter([
            "BRIEF Starting task\n",
            "INFO Task completed\n"
        ])
        mock_process.poll.return_value = 0
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process
        
        # 1. 提交任务
        response = client.post('/api/run', json={
            "data_src": "/tmp/test",
            "max_iter": 1,
            "verbosity": "1",
            "config_path": "test.yaml"
        })
        
        assert response.status_code == 200
        data = response.json
        task_id = data["task_id"]
        assert data["position"] == 0
        
        # 2. 检查队列状态
        response = client.get(f'/api/queue/status/{task_id}')
        status = response.json
        assert status["status"] in ["queued", "running"]
        
        # 3. 等待任务开始
        time.sleep(2)
        
        # 4. 获取日志
        response = client.get('/api/logs', query_string={"run_id": task_id})
        logs = response.json["lines"]
        assert len(logs) > 0
        
        # 5. 检查任务完成
        response = client.get('/api/status', query_string={"run_id": task_id})
        assert response.json["finished"] is True
