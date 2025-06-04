import logging
import os
import random
from collections import defaultdict

from ..prompts import PythonReaderPrompt
from .base_agent import BaseAgent
from .executer_agent import ExecuterAgent
from .utils import init_llm

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
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


class DataPerceptionAgent(BaseAgent):
    """
    Generate the data context using LLM for file content reading.

    Agent Input:
        input_data_folder: Path to the folder to analyze
        max_chars_per_file: Maximum characters per file content

    Agent Output:
        str: Generated the data prompt
    """

    def __init__(self, config, manager, input_data_folder, reader_llm_config, reader_prompt_template):
        super().__init__(config=config, manager=manager)
        self.input_data_folder = input_data_folder
        self.max_chars_per_file = self.config.max_chars_per_file
        self.max_file_group_size_to_show = self.config.max_file_group_size_to_show
        self.num_example_files_to_show = self.config.num_example_files_to_show

        self.language = "python"

        self.reader_llm_config = reader_llm_config
        self.reader_prompt_template = reader_prompt_template

        if self.reader_llm_config.multi_turn:
            self.reader_llm = init_llm(
                llm_config=self.reader_llm_config,
                agent_name=f"{self.language}_reader",
                multi_turn=self.reader_llm_config.multi_turn,
            )

        self.python_reader_prompt = PythonReaderPrompt(
            llm_config=self.reader_llm_config, manager=self.manager, template=self.reader_prompt_template
        )

        self.executer = ExecuterAgent(
            config=self.config,
            manager=self.manager,
            language="python",
            timeout=60,  # TODO: make it configurable
            executer_llm_config=config.executer,
            executer_prompt_template=None,
        )

    def read_file(self, file_path, max_chars):
        # 0. init llm
        if not self.reader_llm_config.multi_turn:
            self.reader_llm = init_llm(
                llm_config=self.reader_llm_config,
                agent_name=f"{self.language}_reader",
                multi_turn=self.reader_llm_config.multi_turn,
            )

        # 1. generate prompt
        prompt = self.python_reader_prompt.build(file_path=file_path, max_chars=max_chars)

        # 2. generate code
        response = self.reader_llm.assistant_chat(prompt)
        generated_python_code = self.python_reader_prompt.parse(response)

        # 3. execute code
        # TODO: add iterative calls if failed
        planner_decision, planner_error_summary, planner_prompt, stderr, stdout = self.executer(
            code_to_execute=generated_python_code,
            code_to_analyze=generated_python_code,
            task_description=prompt,  # use reader's task
            data_prompt=f"file location: {file_path}",
        )

        if stdout:
            result = stdout
            # Truncate if too long
            if len(result) > max_chars:
                result = result[: max_chars - 3] + "..."
        else:
            logger.error(f"Error reading file {file_path}: {stderr}")
            result = f"Error reading file: {stderr}"

        return result

    def __call__(
        self,
    ):
        self.manager.log_agent_start("DataPerceptionAgent: beginning to scan data folder and group similar files.")

        # Get absolute path of the folder
        abs_folder_path = os.path.abspath(self.input_data_folder)
        logger.brief(f"Analyzing folder: {abs_folder_path}")

        # Get list of all files recursively
        all_files = get_all_files(abs_folder_path)
        logger.brief(f"Found {len(all_files)} files")

        # Group similar files
        file_groups = group_similar_files(all_files)
        logger.brief(f"Grouped into {len(file_groups)} patterns")

        # Process files based on their groups and types
        file_contents = {}
        for pattern, group_files in file_groups.items():
            pattern_path = pattern_to_path(pattern, abs_folder_path)
            logger.info(f"Processing pattern: {pattern_path} ({len(group_files)} files)")

            # TODO: ask LLM to decide if we want to show all examples or just one representitive.
            if len(group_files) > self.max_file_group_size_to_show:
                # For large groups, show specified number of examples
                num_examples = min(self.num_example_files_to_show, len(group_files))
                example_files = random.sample(group_files, num_examples)

                group_info = f"Group pattern: {pattern_path} (total {len(group_files)} files)\nExample files:"

                example_contents = []
                for rel_path, abs_path in example_files:
                    logger.brief(f"Reading example file: {abs_path}")
                    content = self.read_file(file_path=abs_path, max_chars=self.max_chars_per_file)
                    example_contents.append(f"Absolute path: {abs_path}\nContent:\n{content}")

                file_contents[group_info] = "\n" + ("-" * 5) + "\n".join(example_contents)
            else:
                # For small groups, show all files
                for rel_path, abs_path in group_files:
                    file_info = f"Absolute path: {abs_path}"

                    # Use LLM to read file content
                    logger.brief(f"Reading file: {abs_path}")

                    file_contents[file_info] = self.read_file(file_path=abs_path, max_chars=self.max_chars_per_file)

        # Generate the prompt
        prompt = f"Absolute path to the folder: {abs_folder_path}\n\nFiles structures:\n\n{'-' * 10}\n\n"
        for file_info, content in file_contents.items():
            prompt += f"{file_info}\nContent:\n{content}\n{'-' * 10}\n"

        self.manager.log_agent_end("DataPerceptionAgent: completed folder scan and assembled data prompt.")

        return prompt
