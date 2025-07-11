"""
File handler for MCP server - manages file uploads and downloads
"""

import base64
import logging
import uuid
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class FileHandler:
    """Handles file upload and download operations"""

    def __init__(self, base_dir: Path):
        """
        Initialize file handler.

        Args:
            base_dir: Base directory for storing uploaded files
        """
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def upload_folder(self, folder_structure: dict, file_contents: dict) -> str:
        """
        Upload a folder with its complete structure and contents.

        Args:
            folder_structure: Directory structure in JSON format
            file_contents: Dict with relative paths as keys and base64 content as values

        Returns:
            str: Server path where folder was uploaded
        """
        # Generate unique folder name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        upload_dir = self.base_dir / f"upload_{timestamp}_{unique_id}"
        upload_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Create folder structure
            self._create_folder_structure(folder_structure, upload_dir)

            # Write files
            for rel_path, content_b64 in file_contents.items():
                file_path = upload_dir / rel_path

                # Ensure parent directory exists
                file_path.parent.mkdir(parents=True, exist_ok=True)

                # Decode and write content
                try:
                    content = base64.b64decode(content_b64)
                    file_path.write_bytes(content)
                    logger.debug(f"Wrote file: {file_path}")
                except Exception as e:
                    logger.error(f"Failed to write file {rel_path}: {str(e)}")
                    raise

            logger.info(f"Successfully uploaded folder to: {upload_dir}")
            return str(upload_dir)

        except Exception as e:
            # Clean up on failure
            if upload_dir.exists():
                import shutil

                shutil.rmtree(upload_dir)
            raise Exception(f"Failed to upload folder: {str(e)}")

    def upload_single_file(self, filename: str, content_b64: str) -> str:
        """
        Upload a single file.

        Args:
            filename: Name of the file
            content_b64: Base64 encoded content

        Returns:
            str: Server path where file was uploaded
        """
        # Generate unique directory for this upload
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        upload_dir = self.base_dir / f"config_{timestamp}_{unique_id}"
        upload_dir.mkdir(parents=True, exist_ok=True)

        file_path = upload_dir / filename

        try:
            # Decode and write content
            content = base64.b64decode(content_b64)
            file_path.write_bytes(content)

            logger.info(f"Successfully uploaded file to: {file_path}")
            return str(file_path)

        except Exception as e:
            # Clean up on failure
            if upload_dir.exists():
                import shutil

                shutil.rmtree(upload_dir)
            raise Exception(f"Failed to upload file: {str(e)}")

    def download_file(self, file_path: str) -> str:
        """
        Download a file from server.

        Args:
            file_path: Server path to the file

        Returns:
            str: Base64 encoded content
        """
        path = Path(file_path)

        # Security check - ensure path is within allowed directories
        if not self._is_safe_path(path):
            raise Exception(f"Access denied: {file_path}")

        if not path.exists():
            raise Exception(f"File not found: {file_path}")

        if not path.is_file():
            raise Exception(f"Not a file: {file_path}")

        try:
            # Read and encode content
            content = path.read_bytes()
            content_b64 = base64.b64encode(content).decode("utf-8")

            logger.info(f"Successfully downloaded file: {file_path}")
            return content_b64

        except Exception as e:
            raise Exception(f"Failed to download file: {str(e)}")

    def list_files(self, directory: str) -> list:
        """
        List all files in a directory recursively.

        Args:
            directory: Server path to directory

        Returns:
            list: List of file paths relative to the directory
        """
        dir_path = Path(directory)

        # Security check
        if not self._is_safe_path(dir_path):
            raise Exception(f"Access denied: {directory}")

        if not dir_path.exists():
            raise Exception(f"Directory not found: {directory}")

        if not dir_path.is_dir():
            raise Exception(f"Not a directory: {directory}")

        files = []
        for path in dir_path.rglob("*"):
            if path.is_file():
                rel_path = path.relative_to(dir_path)
                files.append(str(rel_path))

        return sorted(files)

    def _create_folder_structure(self, structure: dict, base_path: Path):
        """
        Recursively create folder structure.

        Args:
            structure: Folder structure dict
            base_path: Base path to create structure in
        """
        if structure["type"] == "directory":
            # Create directory if it has a name (not root)
            if structure["name"] != "root":
                base_path = base_path / structure["name"]
                base_path.mkdir(parents=True, exist_ok=True)

            # Process children
            for child in structure.get("children", []):
                self._create_folder_structure(child, base_path)

    def _is_safe_path(self, path: Path) -> bool:
        """
        Check if path is safe to access.

        Args:
            path: Path to check

        Returns:
            bool: True if path is safe
        """
        # Convert to absolute path
        abs_path = path.resolve()

        # Check if path is within allowed directories
        allowed_dirs = [
            self.base_dir,
            Path.home() / ".autogluon_assistant",
            Path.cwd() / "runs",  # Default AutoGluon output directory
            Path.cwd() / "mcp",  # MCP output directory
            Path("/tmp") / "autogluon_mcp",  # Temporary directory
        ]

        for allowed_dir in allowed_dirs:
            try:
                # Check if path is within allowed directory
                abs_path.relative_to(allowed_dir.resolve())
                return True
            except ValueError:
                continue

        return False


# Utility functions for client-side file handling


def analyze_folder(folder_path: str) -> dict:
    """
    Analyze folder structure for upload.

    Args:
        folder_path: Path to folder to analyze

    Returns:
        dict: Folder structure representation
    """
    path = Path(folder_path)
    if not path.exists():
        raise Exception(f"Path not found: {folder_path}")

    if not path.is_dir():
        raise Exception(f"Not a directory: {folder_path}")

    def _build_structure(p: Path, root_path: Path) -> dict:
        if p.is_file():
            rel_path = p.relative_to(root_path)
            return {"type": "file", "name": p.name, "size": p.stat().st_size, "path": str(rel_path)}
        else:
            children = []
            for child in sorted(p.iterdir()):
                # Skip hidden files and __pycache__
                if child.name.startswith(".") or child.name == "__pycache__":
                    continue
                children.append(_build_structure(child, root_path))

            return {"type": "directory", "name": p.name if p != root_path else "root", "children": children}

    return _build_structure(path, path)


def read_files_for_upload(folder_path: str, max_file_size: int = 100 * 1024 * 1024) -> dict:
    """
    Read all files in folder and encode for upload.

    Args:
        folder_path: Path to folder
        max_file_size: Maximum file size in bytes (default: 100MB)

    Returns:
        dict: Dict with relative paths as keys and base64 content as values
    """
    path = Path(folder_path)
    if not path.exists() or not path.is_dir():
        raise Exception(f"Invalid directory: {folder_path}")

    file_contents = {}

    for file_path in path.rglob("*"):
        if file_path.is_file():
            # Skip hidden files and __pycache__
            if any(part.startswith(".") for part in file_path.parts):
                continue
            if "__pycache__" in file_path.parts:
                continue

            # Check file size
            if file_path.stat().st_size > max_file_size:
                logger.warning(f"Skipping large file: {file_path} (>{max_file_size} bytes)")
                continue

            try:
                # Read and encode file
                content = file_path.read_bytes()
                content_b64 = base64.b64encode(content).decode("utf-8")

                # Store with relative path
                rel_path = file_path.relative_to(path)
                file_contents[str(rel_path)] = content_b64

            except Exception as e:
                logger.error(f"Failed to read file {file_path}: {str(e)}")

    return file_contents
