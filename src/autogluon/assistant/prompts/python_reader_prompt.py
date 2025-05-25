import logging
import os
from typing import Dict, Optional, Tuple

from .base_prompt import BasePrompt
from .utils import extract_code

logger = logging.getLogger(__name__)


class PythonReaderPrompt(BasePrompt):
    """Handles prompts for code execution evaluation"""

    def default_template(self) -> str:
        return """
Generate Python code to read and analyze the file: "{file_path}"

File Size: {file_size_mb:.2f} MB

Your code should:
1. Import all modules used (e.g. import os).
1. Use appropriate libraries based on file type (pandas for tabular data, etc.)
2. For tabular files (csv, excel, parquet, etc.):
    - Display column names. If there are more than 20 columns, only display the first and last 10.
    - Show first 2-3 rows with truncated cell content (50 chars).
    - Do not show additional index column if it's not in the original table
    - If failed to open the file, treat it as text file
3. For text files:
    - Display first few lines (up to {max_chars} characters)
4. For compressed tabular or text files, show its decompressed content as described.
5. For binary or other files, provide only the most basic information.
6. Keep the total output under {max_chars} characters

Return ONLY the Python code, no explanations. The code should be self-contained and executable on its own.
"""

    def build(self, file_path, max_chars) -> str:
        """Build a prompt for the LLM to evaluate execution logs."""

        file_size = os.path.getsize(file_path)
        file_size_mb = file_size / (1024 * 1024)

        # Format the prompt using the template
        prompt = self.template.format(
            file_path=file_path,
            file_size_mb=file_size_mb,
            max_chars=max_chars,
        )

        # Add format instruction if configured
        if self.llm_config.add_coding_format_instruction:
            format_instruction = (
                "Please format your response with the code in a ```python``` code block to make it easily extractable."
            )
            prompt = f"{prompt}\n\n{format_instruction}"

        self.manager.save_and_log_states(
            content=prompt, save_name="python_reader_prompt.txt", per_iteration=True, add_uuid=True
        )

        return prompt

    def parse(self, response: Dict) -> Tuple[str, Optional[str]]:
        """Parse the LLM's response to generated python code"""

        python_reader_code = extract_code(response=response, language="python")

        self.manager.save_and_log_states(
            content=response, save_name="python_reader_response.txt", per_iteration=True, add_uuid=True
        )
        self.manager.save_and_log_states(
            content=python_reader_code, save_name="python_reader_code.py", per_iteration=True, add_uuid=True
        )

        return python_reader_code
