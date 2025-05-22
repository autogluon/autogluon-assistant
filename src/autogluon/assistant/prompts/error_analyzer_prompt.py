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

    def build(self, prompt_generator) -> str:
        """Build a prompt for the LLM to analyze errors."""

        # Format the prompt using the template
        return self.template.format(
            error_message=prompt_generator.previous_error_message,
            task_description=prompt_generator.task_description,
            data_prompt=prompt_generator.data_prompt,
            user_input=prompt_generator.user_input,
            python_code=prompt_generator.previous_python_code,
            bash_script=prompt_generator.previous_bash_script,
            tutorial_prompt=prompt_generator.previous_tutorial_prompt,
        )

    def parse(self, response: str) -> Optional[str]:
        analysis_match = re.search(r"ERROR_SUMMARY:\s*(.*)", response, re.DOTALL)
        error_analysis
        if analysis_match:
            error_analysis = analysis_match.group(1).strip()
        else:
            error_analysis = "Failed to extract error analysis from LLM response."
        return error_analysis
