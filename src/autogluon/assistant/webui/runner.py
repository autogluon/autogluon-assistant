#!/usr/bin/env python
"""Runner script for streamlit frontend."""

import os
import sys
import subprocess
from pathlib import Path
import argparse


def run_frontend():
    """Run the streamlit frontend application."""
    parser = argparse.ArgumentParser(description="Run AutoGluon Assistant Frontend")
    parser.add_argument(
        "--port",
        type=int,
        default=8509,
        help="Port to run the frontend on (default: 8509)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host to run the frontend on (default: localhost)"
    )
    parser.add_argument(
        "--theme",
        type=str,
        choices=["light", "dark"],
        default=None,
        help="Streamlit theme (default: None)"
    )
    
    args = parser.parse_args()
    
    # 获取 Home.py 的路径
    current_dir = Path(__file__).parent
    home_py_path = current_dir / "Home.py"
    
    # 确保文件存在
    if not home_py_path.exists():
        print(f"Error: {home_py_path} not found!")
        sys.exit(1)
    
    # 构建 streamlit 命令
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(home_py_path),
        f"--server.port={args.port}",
        f"--server.address={args.host}",
    ]
    
    # 添加主题配置（如果指定）
    if args.theme:
        cmd.extend([f"--theme.base={args.theme}"])
    
    # 设置环境变量（如果需要）
    env = os.environ.copy()
    
    try:
        # 运行 streamlit
        print(f"Starting AutoGluon Assistant Frontend on http://{args.host}:{args.port}")
        subprocess.run(cmd, env=env)
    except KeyboardInterrupt:
        print("\nShutting down frontend...")
    except Exception as e:
        print(f"Error running frontend: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_frontend()