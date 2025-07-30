# tests/unittests/mcp/test_local_integration.py

import asyncio
import json
import os
import socket
import subprocess
import threading
import time
from pathlib import Path

import aiohttp
import pytest
import pytest_asyncio
from fastmcp import Client


def log_output(process, service_name):
    """Helper function to log process output in real time"""

    def _log_stream(stream, prefix):
        try:
            for line in iter(stream.readline, b""):
                if line:
                    decoded_line = line.decode("utf-8", errors="replace").strip()
                    if decoded_line:  # Only print non-empty lines
                        print(f"[{service_name}] {prefix}: {decoded_line}")
        except Exception as e:
            print(f"[{service_name}] Error reading {prefix}: {e}")

    # Start threads to read stdout and stderr
    stdout_thread = threading.Thread(target=_log_stream, args=(process.stdout, "OUT"))
    stderr_thread = threading.Thread(target=_log_stream, args=(process.stderr, "ERR"))
    stdout_thread.daemon = True
    stderr_thread.daemon = True
    stdout_thread.start()
    stderr_thread.start()


class TestLocalMCPIntegration:
    """Test complete MCP flow in local setup"""

    @pytest.fixture
    def find_free_port(self):
        """Find available port"""

        def _find_free_port():
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", 0))
                s.listen(1)
                port = s.getsockname()[1]
            return port

        return _find_free_port

    @pytest.fixture
    def test_data(self, tmp_path):
        """Prepare test data"""
        print("\n=== Preparing test data ===")
        # Create input directory
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        print(f"Created input directory: {input_dir}")

        # Create small dataset for faster testing
        print("Creating small regression dataset...")

        # Create train.csv with abalone data
        train_data = """Sex,Length,Diameter,Height,Whole_weight,Shucked_weight,Viscera_weight,Shell_weight,Class_number_of_rings
M,0.62,0.47,0.145,1.0865,0.511,0.2715,0.2565,10
I,0.395,0.29,0.095,0.304,0.127,0.084,0.077,6
I,0.44,0.325,0.1,0.4165,0.185,0.0865,0.11,6
I,0.405,0.3,0.085,0.3035,0.15,0.0505,0.088,7
M,0.68,0.54,0.155,1.534,0.671,0.379,0.384,10
F,0.68,0.56,0.195,1.7775,0.861,0.322,0.415,11
I,0.48,0.365,0.1,0.461,0.2205,0.0835,0.135,8
M,0.545,0.39,0.135,0.7835,0.4225,0.1815,0.156,7
F,0.435,0.35,0.105,0.4195,0.194,0.1005,0.13,7
I,0.245,0.19,0.06,0.086,0.042,0.014,0.025,4
F,0.615,0.455,0.135,1.059,0.4735,0.263,0.274,9
F,0.595,0.46,0.155,1.0455,0.4565,0.24,0.3085,10
M,0.595,0.48,0.165,1.262,0.4835,0.283,0.41,17
F,0.605,0.495,0.17,1.2385,0.528,0.2465,0.39,14
I,0.335,0.25,0.08,0.1695,0.0695,0.044,0.0495,6
F,0.53,0.435,0.17,0.8155,0.2985,0.155,0.275,13
M,0.54,0.405,0.155,0.9715,0.3225,0.194,0.29,19
M,0.615,0.455,0.13,0.9685,0.49,0.182,0.2655,10"""

        (input_dir / "train.csv").write_text(train_data)
        print("Created train.csv with small dataset")

        # Create description file
        description = "Regression on Class_number_of_rings."
        (input_dir / "description.txt").write_text(description)
        print("Created description.txt")

        # Create output directory
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        print(f"Created output directory: {output_dir}")

        return {"input_dir": str(input_dir), "output_dir": str(output_dir)}

    @pytest_asyncio.fixture
    async def services(self, find_free_port):
        """Start all services"""
        print("\n=== Starting services ===")
        processes = []

        # First, stop any existing services
        print("Stopping any existing services...")
        PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
        stop_script = PROJECT_ROOT / "src/autogluon/mcp/server/stop_services.sh"
        if os.path.exists(stop_script):
            subprocess.run([stop_script], capture_output=True)
            await asyncio.sleep(2)

        # Check if default Flask port is available
        flask_port = 5000
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", flask_port))
                s.close()
        except OSError:
            # Port is in use, skip this test
            pytest.skip(f"Port {flask_port} is already in use. Please stop other services first.")

        mcp_server_port = find_free_port()
        mcp_client_port = find_free_port()

        print("Port assignments:")
        print(f"  Flask backend: {flask_port}")
        print(f"  MCP server: {mcp_server_port}")
        print(f"  MCP client: {mcp_client_port}")

        try:
            # Reset the queue database before starting
            print("\n--- Resetting queue database ---")
            PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
            queuedb_script = PROJECT_ROOT / "src/autogluon/assistant/webui/backend/queue/queuedb"
            reset_result = subprocess.run([queuedb_script, "reset"], capture_output=True, text=True)
            if reset_result.returncode == 0:
                print(f"Queue database reset: {reset_result.stdout.strip()}")
            else:
                print(f"Failed to reset queue database: {reset_result.stderr}")

            # 1. Start Flask backend directly on default port
            print("\n--- Starting Flask backend ---")
            flask_env = os.environ.copy()
            flask_env["FLASK_RUN_PORT"] = str(flask_port)
            flask_cmd = ["python", "-m", "autogluon.assistant.webui.backend.app"]
            print(f"Command: {' '.join(flask_cmd)}")

            flask_process = subprocess.Popen(flask_cmd, env=flask_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            processes.append(("Flask", flask_process))
            log_output(flask_process, "Flask")

            # Wait for Flask to start and check it's ready
            print("Waiting for Flask to be ready...")
            flask_ready = False
            for i in range(15):  # Try for 15 seconds
                await asyncio.sleep(1)
                print(f"  Checking Flask health (attempt {i+1}/15)...")
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(f"http://localhost:{flask_port}/api/queue/info") as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                print(f"  Flask is ready! Queue info: {data}")
                                flask_ready = True
                                break
                except Exception as e:
                    print(f"  Flask not ready yet: {str(e)}")

            if not flask_ready:
                raise Exception(f"Flask backend failed to start on port {flask_port}")

            # 2. Start MCP server (directly without the shell script)
            print("\n--- Starting MCP server ---")
            mcp_server_cmd = ["python", "-m", "autogluon.mcp.server.server", "--port", str(mcp_server_port)]
            print(f"Command: {' '.join(mcp_server_cmd)}")

            mcp_server_process = subprocess.Popen(mcp_server_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            processes.append(("MCP-Server", mcp_server_process))
            log_output(mcp_server_process, "MCP-Server")

            # Wait for MCP server to start
            print("Waiting for MCP server to start...")
            await asyncio.sleep(3)

            # 3. Start MCP client
            print("\n--- Starting MCP client ---")
            mcp_client_cmd = ["mlzero-mcp-client", "-p", str(mcp_client_port)]
            print(f"Command: {' '.join(mcp_client_cmd)}")

            mcp_client_process = subprocess.Popen(mcp_client_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            processes.append(("MCP-Client", mcp_client_process))
            log_output(mcp_client_process, "MCP-Client")

            # Wait for MCP client to start
            print("Waiting for MCP client to start...")
            await asyncio.sleep(5)  # Increased wait time

            # Check MCP client health
            print("Checking MCP client health...")
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"http://localhost:{mcp_client_port}/mcp/") as resp:
                        if resp.status == 200:
                            print("  MCP client is ready!")
                        else:
                            print(f"  MCP client returned status {resp.status}")
            except Exception as e:
                print(f"  Failed to check MCP client health: {e}")

            # Check if all services are running
            print("\n--- Checking service status ---")
            for name, p in processes:
                if p.poll() is not None:
                    print(f"ERROR: {name} has exited with code {p.poll()}")
                    raise Exception(f"{name} failed to start")
                else:
                    print(f"✓ {name} is running (PID: {p.pid})")

            print("\n=== All services started successfully ===\n")

            yield {
                "flask_port": flask_port,
                "mcp_server_port": mcp_server_port,
                "mcp_client_port": mcp_client_port,
                "processes": processes,
            }

        finally:
            # Cleanup: terminate all processes
            print("\n=== Cleaning up services ===")
            for name, p in processes:
                if p.poll() is None:
                    print(f"Terminating {name} (PID: {p.pid})...")
                    p.terminate()
                    try:
                        p.wait(timeout=5)
                        print(f"  {name} terminated gracefully")
                    except subprocess.TimeoutExpired:
                        print(f"  {name} did not terminate, killing...")
                        p.kill()

    @pytest.mark.asyncio
    async def test_complete_flow(self, services, test_data):
        """Test complete MCP flow from submission to results"""
        print("\n=== Starting MCP test flow ===")

        # Set timeout for the entire test
        timeout = 600

        async def run_test():
            # Connect to MCP client
            mcp_url = f"http://localhost:{services['mcp_client_port']}/mcp"
            print(f"\nConnecting to MCP client at: {mcp_url}")

            try:
                async with Client(mcp_url) as client:
                    print("Successfully connected to MCP client")

                    # List available tools
                    print("\nListing available tools...")
                    tools = await client.list_tools()
                    print(f"Available tools: {[tool.name for tool in tools]}")

                    # Call run_autogluon_assistant
                    print("\nCalling run_autogluon_assistant with parameters:")
                    params = {
                        "input_folder": test_data["input_dir"],
                        "output_folder": test_data["output_dir"],
                        "server_url": f"http://localhost:{services['mcp_server_port']}/mcp/",  # Added /mcp/ back
                        "verbosity": "detail",  # Changed to detail for more logs
                        "max_iterations": 1,
                        "cleanup_server": True,  # Clean up server files after download
                    }
                    for k, v in params.items():
                        print(f"  {k}: {v}")

                    print("\nSending request...")
                    start_time = time.time()
                    result = await client.call_tool("run_autogluon_assistant", params)
                    elapsed_time = time.time() - start_time

                    # Parse response
                    print(f"\nReceived response after {elapsed_time:.2f} seconds")
                    print(f"Response type: {type(result)}")

                    if hasattr(result, 'content'):
                        response = result.content
                        print(f"Response content type: {type(response)}")
                        print(f"Response content: {response}")
                        
                        if isinstance(response, list) and len(response) > 0:
                            response = response[0]
                            if hasattr(response, 'text'):
                                response = response.text
                        elif isinstance(response, str):
                            pass
                        else:
                            response = str(response)
                    else:
                        response = str(result)

                    print(f"Final response: {response[:200]}...")

                    data = json.loads(response)

                    # Give some time for logs to be printed
                    await asyncio.sleep(1)

                    print(f"\nParsed response: {json.dumps(data, indent=2)}")

                    # Check if there's an error
                    if not data.get("success", False):
                        print(f"\nTask failed with error: {data.get('error', 'Unknown error')}")
                        if "logs" in data:
                            print("\n--- Task logs ---")
                            for log in data["logs"]:
                                print(f"[{log.get('level', 'INFO')}] {log.get('text', '')}")

                    # Verify success
                    assert data["success"] is True, f"Task failed: {data.get('error', 'Unknown error')}"
                    assert "task_id" in data
                    assert "output_directory" in data

                    print("\nTask completed successfully!")
                    print(f"Task ID: {data['task_id']}")
                    print(f"Output directory: {data['output_directory']}")

                    # The output_directory returned is already the mlzero-* directory
                    mlzero_dir = Path(data["output_directory"])
                    assert mlzero_dir.exists(), f"Output directory not found: {mlzero_dir}"
                    print(f"\nChecking output directory: {mlzero_dir}")

                    # Verify critical files exist
                    logs_file = mlzero_dir / "logs.txt"
                    assert logs_file.exists(), f"logs.txt not found in {mlzero_dir}"
                    print("✓ Found logs.txt")

                    # Check generation_iter_0 directory
                    gen_iter_dir = mlzero_dir / "generation_iter_0"
                    assert (
                        gen_iter_dir.exists() and gen_iter_dir.is_dir()
                    ), f"generation_iter_0 directory not found in {mlzero_dir}"
                    print("✓ Found generation_iter_0 directory")

                    # List files in generation_iter_0
                    print("\nFiles in generation_iter_0:")
                    for file in gen_iter_dir.rglob("*"):
                        if file.is_file():
                            print(f"  {file.relative_to(gen_iter_dir)}")

                    print("\n✓ All validations passed!")

            except Exception as e:
                print(f"\nFailed to connect to MCP client: {e}")
                import traceback

                traceback.print_exc()
                raise

        # Run with timeout
        try:
            await asyncio.wait_for(run_test(), timeout=timeout)
        except asyncio.TimeoutError:
            print(f"\n!!! Test timed out after {timeout} seconds")
            print("This might indicate the task is stuck in a loop")
            # Print queue status
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"http://localhost:{services['flask_port']}/api/queue/info") as resp:
                        if resp.status == 200:
                            queue_info = await resp.json()
                            print(f"Final queue status: {queue_info}")
            except:
                pass
            pytest.fail(f"Test timed out after {timeout} seconds")
        except Exception as e:
            print(f"\n!!! Test failed with error: {e}")
            raise e
