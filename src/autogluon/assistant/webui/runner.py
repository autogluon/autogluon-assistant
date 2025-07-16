#!/usr/bin/env python
"""Runner script for streamlit frontend."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_frontend():
    """Run the Streamlit frontend application."""
    parser = argparse.ArgumentParser(description="Run AutoGluon Assistant Frontend")
    parser.add_argument("--port", type=int, default=8509, help="Port to run the frontend on (default: 8509)")
    parser.add_argument(
        "--host", type=str, default="localhost", help="Host to run the frontend on (default: localhost)"
    )
    parser.add_argument(
        "--theme", type=str, choices=["light", "dark"], default=None, help="Streamlit theme (default: None)"
    )

    args = parser.parse_args()

    current_dir = Path(__file__).parent
    home_py_path = current_dir / "Launch_MLZero.py"

    if not home_py_path.exists():
        print(f"Error: {home_py_path} not found!")
        sys.exit(1)

    # Construct the Streamlit launch command
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(home_py_path),
        f"--server.port={args.port}",
        f"--server.address={args.host}",
    ]

    # Include theme settings if provided
    if args.theme:
        cmd.extend([f"--theme.base={args.theme}"])

    # Prepare environment variables
    env = os.environ.copy()

    try:
        # Start the frontend server
        print(f"Starting AutoGluon Assistant Frontend on http://{args.host}:{args.port}")
        subprocess.run(cmd, env=env)
    except KeyboardInterrupt:
        # Handle keyboard interrupt gracefully
        print("\nShutting down frontend...")
    except Exception as e:
        # Report any errors and exit
        print(f"Error running frontend: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_frontend()
