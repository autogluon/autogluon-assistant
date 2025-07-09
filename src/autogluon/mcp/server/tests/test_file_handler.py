#!/usr/bin/env python3
"""
Complete test suite for FileHandler class

Method Input/Output Documentation:

1. FileHandler.__init__(base_dir: Path)
   - Input: base_dir - Path object for base directory
   - Output: None (creates directory if not exists)
   - Side Effect: Creates base_dir directory

2. FileHandler.upload_folder(folder_structure: dict, file_contents: dict) -> str
   - Input:
     - folder_structure: {"type": "directory", "name": "root", "children": [...]}
     - file_contents: {"relative/path.txt": "base64_content", ...}
   - Output: str - absolute path to uploaded folder
   - Side Effect: Creates folder structure and files on disk

3. FileHandler.upload_single_file(filename: str, content_b64: str) -> str
   - Input:
     - filename: "config.yaml"
     - content_b64: "base64_encoded_content"
   - Output: str - absolute path to uploaded file
   - Side Effect: Creates file on disk

4. FileHandler.download_file(file_path: str) -> str
   - Input: file_path - absolute path to file
   - Output: str - base64 encoded file content
   - Raises: Exception if path unsafe or file not found

5. FileHandler.list_files(directory: str) -> list
   - Input: directory - absolute path to directory
   - Output: list of relative file paths ["file1.txt", "subdir/file2.csv"]
   - Raises: Exception if directory not found or unsafe

6. analyze_folder(folder_path: str) -> dict
   - Input: folder_path - path to analyze
   - Output: folder structure dict
   - Example: {"type": "directory", "name": "root", "children": [...]}

7. read_files_for_upload(folder_path: str, max_file_size: int) -> dict
   - Input:
     - folder_path: path to folder
     - max_file_size: max bytes per file (default 100MB)
   - Output: {"relative/path.txt": "base64_content", ...}
   - Skips: hidden files, __pycache__, files over size limit
"""

import base64
import os
import shutil

# Import the module to test
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from file_handler import FileHandler, analyze_folder, read_files_for_upload


class TestFileHandler(unittest.TestCase):
    """Test cases for FileHandler class"""

    def setUp(self):
        """Create temporary directory for tests"""
        self.temp_dir = tempfile.mkdtemp()
        self.base_dir = Path(self.temp_dir) / "uploads"
        self.handler = FileHandler(self.base_dir)

    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir)

    def test_init(self):
        """Test FileHandler initialization"""
        # Test that base directory is created
        self.assertTrue(self.base_dir.exists())
        self.assertTrue(self.base_dir.is_dir())

        # Test with existing directory
        handler2 = FileHandler(self.base_dir)
        self.assertEqual(handler2.base_dir, self.base_dir)

    def test_upload_folder(self):
        """Test folder upload functionality"""
        # Create test folder structure
        folder_structure = {
            "type": "directory",
            "name": "root",
            "children": [
                {"type": "file", "name": "test.txt", "size": 11, "path": "test.txt"},
                {
                    "type": "directory",
                    "name": "subdir",
                    "children": [{"type": "file", "name": "data.csv", "size": 20, "path": "subdir/data.csv"}],
                },
            ],
        }

        # Create file contents
        file_contents = {
            "test.txt": base64.b64encode(b"Hello World").decode("utf-8"),
            "subdir/data.csv": base64.b64encode(b"col1,col2\n1,2\n3,4\n").decode("utf-8"),
        }

        # Upload folder
        upload_path = self.handler.upload_folder(folder_structure, file_contents)

        # Verify upload
        self.assertTrue(Path(upload_path).exists())
        self.assertTrue((Path(upload_path) / "test.txt").exists())
        self.assertTrue((Path(upload_path) / "subdir" / "data.csv").exists())

        # Verify file contents
        with open(Path(upload_path) / "test.txt", "rb") as f:
            self.assertEqual(f.read(), b"Hello World")

        with open(Path(upload_path) / "subdir" / "data.csv", "rb") as f:
            self.assertEqual(f.read(), b"col1,col2\n1,2\n3,4\n")

    def test_upload_single_file(self):
        """Test single file upload"""
        filename = "config.yaml"
        content = b"model: test\niterations: 5\n"
        content_b64 = base64.b64encode(content).decode("utf-8")

        # Upload file
        file_path = self.handler.upload_single_file(filename, content_b64)

        # Verify upload
        path = Path(file_path)
        self.assertTrue(path.exists())
        self.assertTrue(path.is_file())
        self.assertEqual(path.name, filename)

        # Verify content
        with open(path, "rb") as f:
            self.assertEqual(f.read(), content)

    def test_download_file(self):
        """Test file download"""
        # Create a test file
        test_file = self.base_dir / "test_download.txt"
        test_content = b"Download test content"
        test_file.write_bytes(test_content)

        # Download file
        content_b64 = self.handler.download_file(str(test_file))

        # Verify download
        downloaded_content = base64.b64decode(content_b64)
        self.assertEqual(downloaded_content, test_content)

        # Test error cases
        with self.assertRaises(Exception) as cm:
            self.handler.download_file("/etc/passwd")
        self.assertIn("Access denied", str(cm.exception))

        with self.assertRaises(Exception) as cm:
            self.handler.download_file(str(self.base_dir / "nonexistent.txt"))
        self.assertIn("File not found", str(cm.exception))

    def test_list_files(self):
        """Test listing files in directory"""
        # Create test directory structure
        test_dir = self.base_dir / "list_test"
        test_dir.mkdir()
        (test_dir / "file1.txt").write_text("content1")
        (test_dir / "file2.py").write_text("content2")

        subdir = test_dir / "subdir"
        subdir.mkdir()
        (subdir / "file3.csv").write_text("content3")

        # List files
        files = self.handler.list_files(str(test_dir))

        # Verify results
        self.assertEqual(sorted(files), ["file1.txt", "file2.py", "subdir/file3.csv"])

        # Test error cases
        with self.assertRaises(Exception) as cm:
            self.handler.list_files("/etc")
        self.assertIn("Access denied", str(cm.exception))

        with self.assertRaises(Exception) as cm:
            self.handler.list_files(str(test_dir / "nonexistent"))
        self.assertIn("Directory not found", str(cm.exception))

    def test_is_safe_path(self):
        """Test path safety check"""
        # Safe paths
        self.assertTrue(self.handler._is_safe_path(self.base_dir / "test.txt"))
        self.assertTrue(self.handler._is_safe_path(Path.home() / ".autogluon_assistant" / "file.txt"))

        # Unsafe paths
        self.assertFalse(self.handler._is_safe_path(Path("/etc/passwd")))
        self.assertFalse(self.handler._is_safe_path(Path("/usr/bin/python")))

    def test_create_folder_structure(self):
        """Test folder structure creation"""
        test_base = self.base_dir / "structure_test"
        test_base.mkdir()

        structure = {
            "type": "directory",
            "name": "root",
            "children": [
                {
                    "type": "directory",
                    "name": "dir1",
                    "children": [{"type": "directory", "name": "dir2", "children": []}],
                }
            ],
        }

        self.handler._create_folder_structure(structure, test_base)

        # Verify structure
        self.assertTrue((test_base / "dir1").exists())
        self.assertTrue((test_base / "dir1" / "dir2").exists())


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions"""

    def setUp(self):
        """Create temporary directory for tests"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir)

    def test_analyze_folder(self):
        """Test folder analysis"""
        # Create test structure
        test_dir = Path(self.temp_dir) / "analyze_test"
        test_dir.mkdir()

        (test_dir / "file1.txt").write_text("content1")
        (test_dir / ".hidden").write_text("hidden")

        subdir = test_dir / "subdir"
        subdir.mkdir()
        (subdir / "file2.py").write_text("content2")

        # Analyze folder
        structure = analyze_folder(str(test_dir))

        # Verify structure
        self.assertEqual(structure["type"], "directory")
        self.assertEqual(structure["name"], "root")
        self.assertEqual(len(structure["children"]), 2)  # Hidden file excluded

        # Check files
        file_names = [child["name"] for child in structure["children"]]
        self.assertIn("file1.txt", file_names)
        self.assertIn("subdir", file_names)
        self.assertNotIn(".hidden", file_names)

        # Test error cases
        with self.assertRaises(Exception) as cm:
            analyze_folder("/nonexistent/path")
        self.assertIn("Path not found", str(cm.exception))

        with self.assertRaises(Exception) as cm:
            analyze_folder(str(test_dir / "file1.txt"))
        self.assertIn("Not a directory", str(cm.exception))

    def test_read_files_for_upload(self):
        """Test reading files for upload"""
        # Create test files
        test_dir = Path(self.temp_dir) / "read_test"
        test_dir.mkdir()

        # Normal file
        (test_dir / "normal.txt").write_bytes(b"Normal content")

        # Hidden file (should be skipped)
        (test_dir / ".hidden").write_bytes(b"Hidden")

        # Large file (should be skipped)
        large_file = test_dir / "large.bin"
        large_file.write_bytes(b"x" * 1024 * 1024)  # 1MB

        # File in subdirectory
        subdir = test_dir / "subdir"
        subdir.mkdir()
        (subdir / "sub.txt").write_bytes(b"Sub content")

        # __pycache__ (should be skipped)
        pycache = test_dir / "__pycache__"
        pycache.mkdir()
        (pycache / "test.pyc").write_bytes(b"pyc")

        # Read files with 500KB limit
        file_contents = read_files_for_upload(str(test_dir), max_file_size=500 * 1024)

        # Verify results
        self.assertEqual(len(file_contents), 2)  # Only normal.txt and sub.txt
        self.assertIn("normal.txt", file_contents)
        self.assertIn("subdir/sub.txt", file_contents)
        self.assertNotIn(".hidden", file_contents)
        self.assertNotIn("large.bin", file_contents)
        self.assertNotIn("__pycache__/test.pyc", file_contents)

        # Verify content
        decoded = base64.b64decode(file_contents["normal.txt"])
        self.assertEqual(decoded, b"Normal content")

        # Test error cases
        with self.assertRaises(Exception) as cm:
            read_files_for_upload("/nonexistent")
        self.assertIn("Invalid directory", str(cm.exception))


class TestIntegration(unittest.TestCase):
    """Integration tests combining multiple operations"""

    def setUp(self):
        """Create temporary directory for tests"""
        self.temp_dir = tempfile.mkdtemp()
        self.handler = FileHandler(Path(self.temp_dir) / "uploads")

    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir)

    def test_full_workflow(self):
        """Test complete upload/download workflow"""
        # Create source directory
        source_dir = Path(self.temp_dir) / "source"
        source_dir.mkdir()
        (source_dir / "data.csv").write_text("col1,col2\n1,2\n")
        (source_dir / "config.yaml").write_text("model: test\n")

        subdir = source_dir / "models"
        subdir.mkdir()
        (subdir / "model.pkl").write_bytes(b"model data")

        # Step 1: Analyze folder
        structure = analyze_folder(str(source_dir))

        # Step 2: Read files
        file_contents = read_files_for_upload(str(source_dir))

        # Step 3: Upload folder
        upload_path = self.handler.upload_folder(structure, file_contents)

        # Step 4: List uploaded files
        files = self.handler.list_files(upload_path)
        self.assertEqual(sorted(files), ["config.yaml", "data.csv", "models/model.pkl"])

        # Step 5: Download a file
        csv_path = Path(upload_path) / "data.csv"
        content_b64 = self.handler.download_file(str(csv_path))
        content = base64.b64decode(content_b64)
        self.assertEqual(content.decode("utf-8"), "col1,col2\n1,2\n")

        print("✅ Full workflow test passed!")


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
