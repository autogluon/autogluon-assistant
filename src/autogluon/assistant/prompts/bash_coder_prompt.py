import logging
from typing import Dict, Optional, Tuple

from ..constants import ENV_FOLDER_NAME
from .base_prompt import BasePrompt
from .utils import extract_code

logger = logging.getLogger(__name__)


class BashCoderPrompt(BasePrompt):
    """Handles prompts for code execution evaluation"""

    def default_template(self) -> str:
        return """Generate a minimal bash script that will:
{environment_prompt}
Execute the Python script: {python_file_path}

### Python code in the script:
{current_python}

### Previous Error
{error_prompt}

### Previous failed bash script:
{previous_bash}

Notes:
- Generate a minimal, executable bash script
- Focus on essential commands only
- Handle environment and package only if asked or there were errors
"""

    def build(self) -> str:
        """Build a prompt for the LLM to evaluate execution logs."""

        assert self.manager.time_step >= 0, "run manager.step(user_input) before retriving the prompt"

        environment_prompt = self.get_env_prompt()

        # Format the prompt using the template
        prompt = self.template.format(
            environment_prompt=environment_prompt,
            python_file_path=self.manager.python_file_path,
            current_python=self.manager.python_code,
            error_prompt=self.manager.all_previous_error_prompts,
            previous_bash=self.manager.previous_bash_script,
        )

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
        create_venv = self.manager.config.create_venv
        iteration_folder = self.manager.iteration_folder
        common_env_file = self.manager.common_env_file
        selected_tool_env_file = self.manager.selected_tool_env_file

        env_prompt = f"""
Create and configure a conda environment in "{ENV_FOLDER_NAME}" folder under {iteration_folder}:
 - Python version: 3.11
 - Activate the environment
 - pip install uv
 - uv pip install -r {selected_tool_env_file} -r {common_env_file}"""

        if not create_venv:
            env_prompt += (
                "\n - Do not install or update any package unless there is an error due to the missing package.\n"
            )
        else:
            env_prompt += "\n - Install any packages that are needed in the python script"

        return env_prompt
