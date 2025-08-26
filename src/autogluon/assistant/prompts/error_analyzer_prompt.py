import logging
import re
from typing import Optional

from .base_prompt import BasePrompt

logger = logging.getLogger(__name__)


class ErrorAnalyzerPrompt(BasePrompt):
    """Handles prompts for error analysis"""
    
    @classmethod
    def meta_template(cls) -> str:
        """
        Returns specific instructions for meta-prompting the Error Analyzer template.
        """
        return """
The ErrorAnalyzerPrompt analyzes execution errors and provides concise, actionable feedback on how to fix them.

Considerations for rewriting this template:
1. Focus on clear diagnostic analysis of root causes rather than symptoms
2. Emphasize actionable, specific guidance for error resolution
3. Structure the output to clearly separate error diagnosis from solution steps
4. Include checks for common error patterns specific to the task domain
5. Consider the context of previous errors and solutions when providing recommendations
"""
    
    def default_template(self) -> str:
        """Default template for code execution evaluation"""
        return """
Analyze the error and provide your response in this exact format:

ERROR_SUMMARY: [Brief technical description of the root cause in 1-3 sentences]
SUGGESTED_FIX: [Specific debugging directions in 1-3 sentences without code]

### Error Message
{previous_error_message_truncate_mid_8192}

### Task Description
{task_description}

### Data Structures
{data_prompt}

### User Instructions
{user_input}

### Previous Python Code:
{previous_python_code}

### Previous Bash Script to Execute the Python Code:
{previous_bash_script}

### Relevant Tutorials
{previous_tutorial_prompt}
"""

    def _build(self) -> str:
        """Build a prompt for the LLM to analyze errors."""

        # Render the prompt using the variable provider
        prompt = self.render()

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
