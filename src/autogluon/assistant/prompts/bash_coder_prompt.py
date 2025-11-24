import logging
from typing import Dict, Optional, Tuple

from ..constants import ENV_FOLDER_NAME
from .base_prompt import BasePrompt
from .utils import extract_code

logger = logging.getLogger(__name__)


class BashCoderPrompt(BasePrompt):
    """Handles prompts for code execution evaluation"""

    @classmethod
    def meta_instructions(cls) -> str:
        """
        Returns specific instructions for meta-prompting the Bash coder template.
        """
        return """
The BashCoderPrompt generates Bash scripts that execute Python code and set up the required environment.

Considerations for rewriting this template:
1. Focus on proper environment setup with appropriate Python and package versions
2. Ensure correct handling of environment variables and paths
3. Provide clear error handling and logging for diagnostic purposes
4. Include appropriate checks for dependencies and prerequisites
5. Maintain compatibility with the specific execution environment
"""

    def default_template(self) -> str:
        return """Generate a minimal bash script that will:
{environment_prompt}
Execute the Python script: {python_file_path}

### Python code in the script:
{python_code}

### Previous Error
{all_previous_error_analyses}

### Previous failed bash script:
{previous_bash_script}

Notes:
- Generate a minimal, executable bash script
- Focus on essential commands only
- Handle environment and package only if asked or there were errors
"""

    def _build(self, **kwargs) -> str:
        """Build a prompt for the LLM to evaluate execution logs.

        Args:
            **kwargs: Additional keyword arguments to customize the prompt building process
        """

        assert self.manager.time_step >= 0, "run manager.step(user_input) before retriving the prompt"

        environment_prompt = self.get_env_prompt()

        # Render the prompt using the variable provider with additional variables
        additional_vars = {"environment_prompt": environment_prompt}

        prompt = self.render(additional_vars)

        # Add format instruction if configured
        if self.llm_config.add_coding_format_instruction:
            format_instruction = (
                "Please format your response with the code in a ```bash``` code block to make it easily extractable."
            )
            prompt = f"{prompt}\n\n{format_instruction}"

        self.manager.save_and_log_states(
            content=prompt, save_name="bash_coder_prompt.txt", per_iteration=True, add_uuid=False
        )

        return prompt

    def parse(self, response: Dict) -> Tuple[str, Optional[str]]:
        """Parse the LLM's response to generated bash code"""

        extracted_bash_script = extract_code(response=response, language="bash")

        self.manager.save_and_log_states(
            content=response, save_name="bash_coder_response.txt", per_iteration=True, add_uuid=False
        )
        self.manager.save_and_log_states(
            content=extracted_bash_script, save_name="extracted_bash_script.sh", per_iteration=True, add_uuid=False
        )

        return extracted_bash_script

    def get_env_prompt(self):
        configure_env = self.manager.configure_env
        iteration_folder = self.manager.iteration_folder
        selected_tool = self.manager.selected_tool
        common_env_file = self.manager.common_env_file
        selected_tool_env_file = self.manager.selected_tool_env_file

        env_prompt = f"""
Create and configure a conda environment in "{ENV_FOLDER_NAME}" folder under {iteration_folder}:
 - Python version: 3.11
 - Activate the environment
 - pip install uv
 - Install required packages from {common_env_file} and {selected_tool_env_file} using uv pip install -r {selected_tool_env_file} --prerelease=allow -r {common_env_file}"""

        if not configure_env:
            env_prompt += f"\n - Only install the exact packages specified in the requirements files with their dependencies.\n - Do NOT upgrade or reinstall {selected_tool} if it's already at the correct version specified in the requirements."
        else:
            env_prompt += (
                "\n - Install any additional packages that are needed for the python script to run successfully"
            )

        return env_prompt
