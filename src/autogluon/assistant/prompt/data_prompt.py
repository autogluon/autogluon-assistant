import logging
import os
from collections import defaultdict

# Import the file reader utility
from ..reader import LLMFileReader

# Configure logging
logger = logging.getLogger(__name__)


def get_all_files(folder_path):
    """
    Recursively get all files in the folder and its subfolders.
    Returns a list of tuples containing (relative_path, absolute_path).
    """
    all_files = []
    abs_folder_path = os.path.abspath(folder_path)

    for root, _, files in os.walk(abs_folder_path):
        for file in files:
            abs_path = os.path.join(root, file)
            rel_path = os.path.relpath(abs_path, abs_folder_path)
            all_files.append((rel_path, abs_path))

    return all_files


def group_similar_files(files):
    """
    Group files based on their folder structure and extensions.
    Files are placed in the same group if they follow the same pattern at each level.
    At each level, if there are 5 or fewer unique folders, the actual folder names are used,
    otherwise a wildcard '*' is used.

    Parameters:
    files: List of tuples (relative_path, absolute_path)

    Returns:
    Dict mapping group keys to lists of tuples (relative_path, absolute_path)
    """
    # First, analyze folder counts at each depth level
    depth_folders = defaultdict(set)
    max_depth = 0

    # Collect all unique folders at each depth
    for rel_path, _ in files:
        parts = os.path.normpath(rel_path).split(os.sep)
        max_depth = max(max_depth, len(parts) - 1)  # -1 for filename

        # Record folders at each depth
        for depth, folder in enumerate(parts[:-1]):  # Exclude filename
            depth_folders[depth].add(folder)

    # Create groups
    groups = defaultdict(list)
    for rel_path, abs_path in files:
        parts = os.path.normpath(rel_path).split(os.sep)
        filename = parts[-1]
        folders = parts[:-1]

        # Get file extension (if any)
        ext = os.path.splitext(filename)[1].lower()

        # Build group key parts
        group_key_parts = []

        # Add folder pattern for each depth
        for depth, folder in enumerate(folders):
            unique_folders = depth_folders[depth]
            if len(unique_folders) <= 5:
                # Use actual folder name if 5 or fewer unique folders at this depth
                group_key_parts.append(folder)
            else:
                # Use wildcard if more than 5 unique folders
                group_key_parts.append("*")

        # Add extension pattern
        group_key_parts.append(ext if ext else "NO_EXT")

        # Convert key to immutable tuple for dictionary
        group_key = tuple(group_key_parts)
        groups[group_key].append((rel_path, abs_path))

    return groups


def pattern_to_path(pattern, base_path):
    """
    Convert a group pattern tuple to an absolute path string.
    Pattern tuple format: (folder1, folder2, ..., extension)

    Parameters:
    pattern: Tuple of folder names and extension
    base_path: Base directory path to make the pattern absolute
    """
    # Last element is extension
    folders = pattern[:-1]  # Get all folder patterns
    ext = pattern[-1]

    # Create a path-like string from folder patterns
    path_parts = []
    for folder in folders:
        path_parts.append(str(folder))

    # Add a placeholder filename with the extension
    if ext == "NO_EXT":
        path_parts.append("*")
    else:
        path_parts.append(f"*{ext}")

    # Join with base path to make it absolute
    relative_pattern = os.path.join(*path_parts) if path_parts else "*"
    return os.path.join(base_path, relative_pattern)


def generate_data_prompt_with_llm(input_data_folder, max_chars_per_file, llm_config):
    """
    Generate a data prompt using LLM for file content reading.

    Args:
        input_data_folder: Path to the folder to analyze
        max_chars_per_file: Maximum characters per file content
        llm_reader: LLMFileReader instance

    Returns:
        str: Generated data prompt
    """
    llm_reader = LLMFileReader(llm_config=llm_config)

    # Get absolute path of the folder
    abs_folder_path = os.path.abspath(input_data_folder)
    logger.info(f"Analyzing folder: {abs_folder_path}")

    # Get list of all files recursively
    all_files = get_all_files(abs_folder_path)
    logger.info(f"Found {len(all_files)} files")

    # Group similar files
    file_groups = group_similar_files(all_files)
    logger.info(f"Grouped into {len(file_groups)} patterns")

    # Process files based on their groups and types
    file_contents = {}
    for pattern, group_files in file_groups.items():
        pattern_path = pattern_to_path(pattern, abs_folder_path)
        logger.info(f"Processing pattern: {pattern_path} ({len(group_files)} files)")

        if len(group_files) > 5:  # TODO: ask LLM to decide if we want to show all examples or just one representitive.
            # For large groups, only show one example
            example_rel_path, example_abs_path = group_files[0]
            group_info = f"Group pattern: {pattern_path} (total {len(group_files)} files)\nExample file:\nAbsolute path: {example_abs_path}"

            # Use LLM to read file content
            logger.info(f"Reading example file: {example_abs_path}")
            file_contents[group_info] = llm_reader(file_path=example_abs_path, max_chars=max_chars_per_file)
        else:
            # For small groups, show all files
            for rel_path, abs_path in group_files:
                file_info = f"Absolute path: {abs_path}"

                # Use LLM to read file content
                logger.info(f"Reading file: {abs_path}")
                file_contents[file_info] = llm_reader(file_path=abs_path, max_chars=max_chars_per_file)

    # Generate the prompt
    prompt = f"Absolute path to the folder: {abs_folder_path}\n\nFiles structures:\n\n{'-' * 10}\n\n"
    for file_info, content in file_contents.items():
        prompt += f"{file_info}\nContent:\n{content}\n{'-' * 10}\n"

    return prompt
