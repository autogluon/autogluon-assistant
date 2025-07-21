def main():
    import argparse
    import os
    import subprocess

    parser = argparse.ArgumentParser(description="AutoGluon MCP Server Runner")
    parser.add_argument("--flask-port", "-f", type=int, default=5000, help="Flask backend port (default: 5000)")
    parser.add_argument("--server-port", "-s", type=int, default=8000, help="MCP server port (default: 8000)")
    args = parser.parse_args()

    # Build command with port arguments
    cmd = ["/bin/bash", os.path.join(os.path.dirname(__file__), "start_services.sh")]
    cmd.extend(["--flask-port", str(args.flask_port)])
    cmd.extend(["--server-port", str(args.server_port)])
    
    subprocess.run(cmd)
