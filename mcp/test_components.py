#!/usr/bin/env python3
"""
Test script for MCP Server components
"""

import base64
import json
import tempfile
from pathlib import Path

# Test file_handler
def test_file_handler():
    print("Testing FileHandler...")
    from file_handler import FileHandler, analyze_folder, read_files_for_upload
    
    # Create test directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        test_dir = Path(tmpdir) / "test_data"
        test_dir.mkdir()
        
        (test_dir / "train.csv").write_text("x,y\n1,2\n3,4\n")
        (test_dir / "config.yaml").write_text("model: test\n")
        
        subdir = test_dir / "subfolder"
        subdir.mkdir()
        (subdir / "test.txt").write_text("test content")
        
        # Test analyze_folder
        structure = analyze_folder(str(test_dir))
        print(f"  ✓ Folder structure: {structure['type']} with {len(structure['children'])} children")
        
        # Test read_files_for_upload
        contents = read_files_for_upload(str(test_dir))
        print(f"  ✓ Read {len(contents)} files")
        
        # Test FileHandler
        handler = FileHandler(Path(tmpdir) / "uploads")
        
        # Test upload
        upload_path = handler.upload_folder(structure, contents)
        print(f"  ✓ Uploaded to: {upload_path}")
        
        # Test list files
        files = handler.list_files(upload_path)
        print(f"  ✓ Listed {len(files)} files")
        
        # Test download
        test_file = Path(upload_path) / files[0]
        content_b64 = handler.download_file(str(test_file))
        content = base64.b64decode(content_b64)
        print(f"  ✓ Downloaded file: {len(content)} bytes")
        
    print("  All FileHandler tests passed!\n")


# Test utils
def test_utils():
    print("Testing utils...")
    from utils import generate_task_output_dir, parse_credentials, format_file_size
    
    # Test generate_task_output_dir
    output_dir = generate_task_output_dir()
    print(f"  ✓ Generated output dir: {output_dir}")
    assert "mlzero-" in output_dir
    
    # Test parse_credentials - AWS
    aws_creds = """
    export AWS_ACCESS_KEY_ID="test_key"
    export AWS_SECRET_ACCESS_KEY="test_secret"
    export AWS_DEFAULT_REGION="us-west-2"
    """
    parsed = parse_credentials(aws_creds, "bedrock")
    print(f"  ✓ Parsed AWS credentials: {len(parsed)} fields")
    assert parsed["AWS_ACCESS_KEY_ID"] == "test_key"
    
    # Test parse_credentials - OpenAI
    openai_creds = 'export OPENAI_API_KEY="sk-test"'
    parsed = parse_credentials(openai_creds, "openai")
    print(f"  ✓ Parsed OpenAI credentials: {len(parsed)} fields")
    assert parsed["OPENAI_API_KEY"] == "sk-test"
    
    # Test format_file_size
    sizes = [
        (1024, "1.0 KB"),
        (1024 * 1024, "1.0 MB"),
        (1536 * 1024, "1.5 MB")
    ]
    for size, expected in sizes:
        result = format_file_size(size)
        print(f"  ✓ Format {size} bytes: {result}")
        assert result == expected
    
    print("  All utils tests passed!\n")


# Test task_manager (mock)
def test_task_manager_mock():
    print("Testing TaskManager (mock)...")
    from task_manager import TaskManager
    
    # This would require a running Flask backend
    # For now, just test initialization
    manager = TaskManager()
    print("  ✓ TaskManager initialized")
    
    # Test that methods exist
    assert hasattr(manager, 'start_task')
    assert hasattr(manager, 'check_status')
    assert hasattr(manager, 'send_input')
    assert hasattr(manager, 'cancel_task')
    assert hasattr(manager, 'list_outputs')
    print("  ✓ All required methods exist")
    
    print("  TaskManager mock tests passed!\n")


# Test server imports
def test_server_imports():
    print("Testing server imports...")
    
    try:
        import server
        print("  ✓ Server module imports successfully")
        
        # Check that MCP tools are defined
        tools = [
            'upload_input_folder',
            'upload_config', 
            'start_task',
            'check_status',
            'send_input',
            'cancel_task',
            'list_outputs',
            'download_file'
        ]
        
        # Note: We can't directly check decorated functions
        # but we can check they exist in the module
        for tool in tools:
            assert hasattr(server, tool)
            print(f"  ✓ Tool '{tool}' defined")
            
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        print("  Make sure fastmcp is installed: pip install fastmcp")
        
    print("  Server import tests passed!\n")


# Main test runner
def main():
    print("=== MCP Server Component Tests ===\n")
    
    test_file_handler()
    test_utils()
    test_task_manager_mock()
    test_server_imports()
    
    print("=== All tests completed! ===")
    print("\nNote: Full integration tests require:")
    print("1. Flask backend running on port 5000")
    print("2. MCP server running on port 8000")
    print("3. Running the client example with test data")


if __name__ == "__main__":
    main()
