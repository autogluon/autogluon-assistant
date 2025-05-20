import logging
import os
import subprocess
import sys
import tempfile
from datetime import datetime

from omegaconf import DictConfig

from ..llm import ChatLLMFactory

logger = logging.getLogger(__name__)


class LLMFileReader:
    """
    A utility class that uses LLM to generate code for reading file content
    and executes that code to get the result.
    """

    def __init__(self, llm_config: DictConfig):
        """
        Initialize with an LLM instance.

        Args:
            llm: An initialized LLM that has an assistant_chat method
        """
        self.llm_config = llm_config
        self.multi_turn = llm_config.multi_turn
        self.add_coding_format_instruction = (
            llm_config.add_coding_format_instruction
            if hasattr(self.llm_config, "add_coding_format_instruction")
            else False
        )
        self.details = llm_config.details
        if self.multi_turn:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            self.llm = ChatLLMFactory.get_chat_model(llm_config, session_name=f"multi_round_coder_{timestamp}")

    def generate_reader_code(self, file_path, max_chars=1000):
        """
        Generate Python code to read a specific file and display its content.

        Args:
            file_path: Path to the file to analyze
            max_chars: Maximum character limit for output

        Returns:
            str: Python code to read the file
        """
        file_size = os.path.getsize(file_path)
        file_size_mb = file_size / (1024 * 1024)

        prompt = f"""
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
           {"- Count total rows and provide basic statistics" if self.details else "- No additional info needed."}
        3. For text files:
           - Display first few lines (up to {max_chars} characters)
        4. For compressed tabular or text files, show its decompressed content as described.
        {"5. For other files, provide appropriate summary" if self.details else "4. For binary or other files, provide only file size."}
        6. Keep the total output under {max_chars} characters
        
        Return ONLY the Python code, no explanations or markdown. The code should be self-contained
        and executable on its own.
        """

        # Add format instruction if configured
        if self.add_coding_format_instruction:
            format_instruction = (
                "Please format your response with the code in a ```python``` code block to make it easily extractable."
            )
            prompt = f"{prompt}\n\n{format_instruction}"

        response = self.llm.assistant_chat(prompt)

        # Extract code from potential markdown formatting
        if "```python" in response:
            code_parts = response.split("```python")
            code_block = code_parts[1].split("```")[0].strip()
            return code_block
        elif "```" in response:
            code_parts = response.split("```")
            code_block = code_parts[1].strip()
            return code_block
        else:
            return response.strip()

    def execute_code(self, code, timeout=30):
        """
        Execute the generated Python code and capture the output.

        Args:
            code: Python code to execute
            timeout: Maximum execution time in seconds

        Returns:
            str: Execution result (stdout) or error message
        """
        # Create a temporary Python file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_file = f.name

        try:
            # Execute the code with timeout
            result = subprocess.run([sys.executable, temp_file], capture_output=True, text=True, timeout=timeout)

            # Return stdout if successful, otherwise stderr
            if result.returncode == 0 and result.stdout:
                return result.stdout
            else:
                return f"Error executing code: {result.stderr}"

        except subprocess.TimeoutExpired:
            return f"Error: Code execution timed out after {timeout} seconds"
        except Exception as e:
            return f"Error: {str(e)}"
        finally:
            # Clean up the temporary file
            try:
                os.unlink(temp_file)
            except Exception:
                pass

    def __call__(self, file_path, max_chars=1000, timeout=30):
        """
        Main method to read file content using LLM-generated code.

        Args:
            file_path: Path to the file to read
            max_chars: Maximum character output
            timeout: Maximum execution time in seconds

        Returns:
            str: The file content or error message
        """
        if not self.multi_turn:
            # create a new session if not multi_turn
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            self.llm = ChatLLMFactory.get_chat_model(self.llm_config, session_name=f"single_round_coder_{timestamp}")

        try:
            # Check if file exists
            if not os.path.exists(file_path):
                return f"Error: File not found: {file_path}"

            # Generate code to read the file
            code = self.generate_reader_code(file_path, max_chars)

            # Execute the code and get the result
            result = self.execute_code(code, timeout)

            # Truncate if too long
            if len(result) > max_chars:
                result = result[: max_chars - 3] + "..."

            return result

        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return f"Error reading file: {str(e)}"


# Example usage:
"""
from .llm import ChatLLMFactory

# Initialize LLM
llm_config = {...}  # Your LLM configuration
llm = ChatLLMFactory.get_chat_model(llm_config)

# Create file reader
file_reader = LLMFileReader(llm)

# Read a file
content = file_reader.read_file("/path/to/file.csv", max_chars=2000)
print(content)
"""
