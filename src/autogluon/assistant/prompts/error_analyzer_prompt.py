import logging
import re
from typing import Optional

from .base_prompt import BasePrompt

logger = logging.getLogger(__name__)


class ErrorAnalyzerPrompt(BasePrompt):
    """Handles prompts for error analysis"""

    def default_template(self) -> str:
        """Default template for code execution evaluation"""
        return """
Analyze the error and provide your response in this exact format:

ERROR_SUMMARY: [Brief technical description of the root cause in 1-3 sentences]
SUGGESTED_FIX: [Specific debugging directions in 1-3 sentences without code]

### Error Message
{error_message}

### Task Description
{task_description}

### Data Structures
{data_prompt}

### User Instructions
{user_input}

### Previous Python Code:
{python_code}

### Previous Bash Script to Execute the Python Code:
{bash_script}

### Relevant Tutorials
{tutorial_prompt}
"""

    def build(self) -> str:
        """Build a prompt for the LLM to analyze errors."""

        previous_error_message = self._truncate_output_mid(
            output=self.manager.previous_error_message, max_length=self.manager.config.max_error_message_length
        )

        # Format the prompt using the template
        prompt = self.template.format(
            error_message=previous_error_message,
            task_description=self.manager.task_description,
            data_prompt=self.manager.data_prompt,
            user_input=self.manager.user_input,
            python_code=self.manager.previous_python_code,
            bash_script=self.manager.previous_bash_script,
            tutorial_prompt=self.manager.previous_tutorial_prompt,
        )

        self.manager.save_and_log_states(
            content=prompt, save_name="error_analyzer_prompt.txt", per_iteration=True, add_uuid=False
        )

        return prompt

    def parse(self, response: str) -> Optional[str]:
        analysis_match = re.search(r"ERROR_SUMMARY:\s*(.*)", response, re.DOTALL)
        if analysis_match:
            error_analysis = f"ERROR_SUMMARY: {analysis_match.group(1).strip()}"
        else:
            error_analysis = "Failed to extract error analysis from LLM response."

        self.manager.save_and_log_states(
            content=response, save_name="error_analyzer_response.txt", per_iteration=True, add_uuid=False
        )
        self.manager.save_and_log_states(
            content=error_analysis, save_name="error_analysis.txt", per_iteration=True, add_uuid=False
        )
        return error_analysis
