#!/usr/bin/env python3
"""
Example MCP client for AutoGluon Assistant

This demonstrates how to use the MCP server to:
1. Upload input data
2. Start an AutoGluon task
3. Monitor progress
4. Handle user input if needed
5. Download results
"""

import asyncio
import base64
import json
import sys
from pathlib import Path
from typing import Optional

from fastmcp import Client

# Import utility functions from file_handler
sys.path.append(str(Path(__file__).parent.parent))
from file_handler import analyze_folder, read_files_for_upload


def parse_mcp_response(response):
    """Parse FastMCP response format"""
    if isinstance(response, list) and len(response) > 0:
        # Extract text content from TextContent object
        text_content = response[0].text
        # Parse JSON from text
        return json.loads(text_content)
    return response


async def cleanup_server_files(client, output_dir):
    """Clean up server files after download"""
    if not output_dir:
        print("   No output directory to clean")
        return
    
    print(f"   Cleaning up server directory: {output_dir}")
    result = await client.call_tool("cleanup_output", {
        "output_dir": output_dir
    })
    result = parse_mcp_response(result)
    
    if result.get("success"):
        print("   ✓ Server files cleaned up")
    else:
        print(f"   ✗ Cleanup failed: {result.get('error', 'Unknown error')}")


async def run_autogluon_task(
    input_folder: str,
    output_folder: str,
    config_file: Optional[str] = None,
    max_iterations: int = 5,
    need_user_input: bool = False,
    provider: str = "bedrock",
    model: Optional[str] = None,
    credentials_text: Optional[str] = None,
    server_url: str = "http://localhost:8000",
    cleanup_server: bool = True
):
    """
    Run AutoGluon task via MCP server.
    
    Args:
        input_folder: Local path to input data
        output_folder: Local path where results will be saved
        config_file: Optional path to config file
        max_iterations: Maximum iterations
        need_user_input: Whether to enable interactive mode
        provider: LLM provider
        model: Model name (optional)
        credentials_text: Credentials in environment variable format
        server_url: MCP server URL
    """
    # Create client - add /mcp to the URL
    if not server_url.endswith('/mcp'):
        server_url = server_url.rstrip('/') + '/mcp'
    client = Client(server_url)
    
    async with client:
        print("Connected to AutoGluon MCP Server")
        
        # 1. Upload input folder
        print(f"\n1. Uploading input folder: {input_folder}")
        
        # Analyze folder structure
        folder_structure = analyze_folder(input_folder)
        print(f"   Found {count_files(folder_structure)} files")
        
        # Read file contents
        file_contents = read_files_for_upload(input_folder)
        print(f"   Total size: {sum(len(c) for c in file_contents.values()) / 1024 / 1024:.1f} MB")
        
        # Upload folder
        result = await client.call_tool("upload_input_folder", {
            "folder_structure": folder_structure,
            "file_contents": file_contents
        })
        result = parse_mcp_response(result)
        
        if not result.get("success", False):
            print(f"   ERROR: {result.get('error', 'Upload failed')}")
            return
            
        server_input_dir = result["path"]
        print(f"   ✓ Uploaded to: {server_input_dir}")
        
        # 2. Upload config file if provided
        server_config_path = None
        if config_file:
            print(f"\n2. Uploading config file: {config_file}")
            
            config_path = Path(config_file)
            if not config_path.exists():
                print(f"   ERROR: Config file not found")
                return
                
            # Read and encode config
            config_content = config_path.read_bytes()
            config_b64 = base64.b64encode(config_content).decode('utf-8')
            
            result = await client.call_tool("upload_config", {
                "filename": config_path.name,
                "content": config_b64
            })
            result = parse_mcp_response(result)
            
            if not result["success"]:
                print(f"   ERROR: {result.get('error', 'Upload failed')}")
                return
                
            server_config_path = result["path"]
            print(f"   ✓ Uploaded to: {server_config_path}")
        
        # 3. Start task
        print(f"\n3. Starting AutoGluon task")
        print(f"   Provider: {provider}")
        print(f"   Model: {model or 'default'}")
        print(f"   Max iterations: {max_iterations}")
        print(f"   Interactive mode: {need_user_input}")
        
        result = await client.call_tool("start_task", {
            "input_dir": server_input_dir,
            "output_dir": output_folder,
            "config_path": server_config_path,
            "max_iterations": max_iterations,
            "need_user_input": need_user_input,
            "provider": provider,
            "model": model,
            "credentials_text": credentials_text
        })
        result = parse_mcp_response(result)
        
        if not result["success"]:
            print(f"   ERROR: {result.get('error', 'Failed to start task')}")
            return
            
        task_id = result["task_id"]
        position = result.get("position", 0)
        
        print(f"   ✓ Task started: {task_id}")
        if position > 0:
            print(f"   Queue position: {position}")
        
        # 4. Monitor progress
        print(f"\n4. Monitoring task progress...")
        
        last_log_count = 0
        while True:
            # Check status
            status = await client.call_tool("check_status", {})
            status = parse_mcp_response(status)
            
            if not status["success"]:
                print(f"   ERROR: {status.get('error', 'Status check failed')}")
                break
            
            # Print new logs
            logs = status.get("logs", [])
            new_logs = logs[last_log_count:]
            for log in new_logs:
                if isinstance(log, dict):
                    level = log.get("level", "INFO")
                    text = log.get("text", "")
                    print(f"   [{level}] {text}")
                else:
                    print(f"   {log}")
            last_log_count = len(logs)
            
            # Check if waiting for input
            if status.get("waiting_for_input"):
                prompt = status.get("input_prompt", "Please provide input:")
                print(f"\n   PROMPT: {prompt}")
                user_input = input("   Your input (press Enter to skip): ")
                
                result = await client.call_tool("send_input", {
                    "user_input": user_input
                })
                result = parse_mcp_response(result)
                
                if not result["success"]:
                    print(f"   ERROR: Failed to send input")
                else:
                    print("   ✓ Input sent")
            
            # Check if completed
            if status.get("state") == "completed":
                print("\n   ✓ Task completed successfully!")
                break
            elif status.get("state") == "failed":
                print("\n   ✗ Task failed!")
                break
            
            # Update progress
            progress_info = await client.call_tool("get_progress", {})
            progress_info = parse_mcp_response(progress_info)
            if isinstance(progress_info, dict):
                progress = progress_info.get("progress", 0.0)
                print(f"   Progress: {progress * 100:.1f}%", end="\r")
            
            # Wait before next check
            await asyncio.sleep(2)
        
        # 5. Download results
        print(f"\n5. Downloading results to: {output_folder}")
        
        # List output files
        result = await client.call_tool("list_outputs", {})
        result = parse_mcp_response(result)
        
        if not result["success"]:
            print(f"   ERROR: {result.get('error', 'Failed to list outputs')}")
            return
            
        files = result["files"]
        output_dir = result.get("output_dir")  # Get the server output directory
        print(f"   Found {len(files)} output files")
        print(f"   Server output directory: {output_dir}")
        
        # Create output directory with same structure as server
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Extract the mlzero-xxx folder name from server path
        if output_dir:
            server_folder_name = Path(output_dir).name  # e.g., mlzero-20250623_223850-xxx
            local_output_base = output_path / server_folder_name
            local_output_base.mkdir(exist_ok=True)
        else:
            local_output_base = output_path
        
        # Download each file preserving directory structure
        for file_path in files:
            print(f"   Downloading: {file_path}")
            
            result = await client.call_tool("download_file", {
                "file_path": file_path
            })
            result = parse_mcp_response(result)
            
            if not result["success"]:
                print(f"     ERROR: {result.get('error', 'Download failed')}")
                continue
            
            # Decode and save file
            content = base64.b64decode(result["content"])
            
            # Preserve full directory structure
            if output_dir and file_path.startswith(output_dir):
                # Get relative path from server output directory
                rel_path = Path(file_path).relative_to(output_dir)
            else:
                # Fallback to just filename
                rel_path = Path(file_path).name
            
            local_path = local_output_base / rel_path
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(content)
            
            print(f"     ✓ Saved to: {local_path}")
        
        print(f"\n✓ All done! Results saved to: {local_output_base}")
        
        # Optionally clean up server files
        if cleanup_server and output_dir:
            print("\nClean up server files? (y/n): ", end="")
            if input().lower() == 'y':
                await cleanup_server_files(client, output_dir)


def count_files(structure: dict) -> int:
    """Count files in folder structure"""
    if structure["type"] == "file":
        return 1
    else:
        return sum(count_files(child) for child in structure.get("children", []))


def load_credentials_from_file(file_path: str) -> str:
    """Load credentials from file"""
    path = Path(file_path)
    if path.exists():
        return path.read_text()
    return ""


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="AutoGluon MCP Client Example")
    parser.add_argument("input", help="Input data folder")
    parser.add_argument("output", help="Output folder for results")
    parser.add_argument("-c", "--config", help="Config file path")
    parser.add_argument("-n", "--iterations", type=int, default=5, help="Max iterations")
    parser.add_argument("-i", "--interactive", action="store_true", help="Enable interactive mode")
    parser.add_argument("-p", "--provider", default="bedrock", choices=["bedrock", "openai", "anthropic"])
    parser.add_argument("-m", "--model", help="Model name")
    parser.add_argument("--creds", help="Path to credentials file")
    parser.add_argument("--server", default="http://localhost:8000", help="MCP server URL")
    parser.add_argument("--no-cleanup", action="store_true", help="Don't clean up server files after download")
    
    args = parser.parse_args()
    
    # Load credentials if provided
    credentials_text = None
    if args.creds:
        credentials_text = load_credentials_from_file(args.creds)
        if not credentials_text:
            print(f"Warning: Could not load credentials from {args.creds}")
    
    # Run the task
    asyncio.run(run_autogluon_task(
        input_folder=args.input,
        output_folder=args.output,
        config_file=args.config,
        max_iterations=args.iterations,
        need_user_input=args.interactive,
        provider=args.provider,
        model=args.model,
        credentials_text=credentials_text,
        server_url=args.server,
        cleanup_server=not args.no_cleanup
    ))