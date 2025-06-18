import logging
from typing import Dict, Optional, Tuple

from .base_prompt import BasePrompt

logger = logging.getLogger(__name__)


class ExecuterPrompt(BasePrompt):
    """Handles prompts for code execution evaluation"""

    def default_template(self) -> str:
        """Default template for code execution evaluation"""
        return """You are an expert code evaluator. Analyze the execution results of the following Python code and determine if the execution was successful or if issues need to be fixed.

### Task Descriptions
{task_description}

### Data Structure
{data_prompt}

### Python Code
{python_code}

## Execution Results
### Standard Output (stdout)

{stdout}

### Standard Error (stderr)

{stderr}

Evaluate the execution results and decide on one of the following actions:
1. FINISH - If the execution was completely successful and met all requirements.
2. FIX - If there were errors, issues, or performance problems that need to be addressed.
Provide your decision in the following format:
DECISION: [FINISH or FIX]
ERROR_SUMMARY: [Brief summary of errors if any, or "None" if no errors]
The error summary should be brief but informative enough for another agent to understand what needs to be fixed.
Even if the code executed without throwing errors, it might still have issues with logic or not meet all requirements."""

    def build(self, stdout: str, stderr: str, python_code: str, task_description: str, data_prompt: str) -> str:
        """Build a prompt for the LLM to evaluate execution logs."""
        self.manager.save_and_log_states(content=stdout, save_name="stdout.txt", per_iteration=True, add_uuid=True)
        self.manager.save_and_log_states(content=stderr, save_name="stderr.txt", per_iteration=True, add_uuid=True)

        # Truncate outputs if they exceed max length
        stdout = self._truncate_output_mid(stdout, self.llm_config.max_stdout_length)
        stderr = self._truncate_output_mid(stderr, self.llm_config.max_stderr_length)

        self.manager.save_and_log_states(
            content=stdout, save_name="stdout(truncated).txt", per_iteration=True, add_uuid=True
        )
        self.manager.save_and_log_states(
            content=stderr, save_name="stderr(truncated).txt", per_iteration=True, add_uuid=True
        )

        # Format the prompt using the template
        prompt = self.template.format(
            task_description=task_description,
            data_prompt=data_prompt,
            python_code=python_code,
            stdout=stdout or "No standard output",
            stderr=stderr or "No standard error",
        )

        self.manager.save_and_log_states(
            content=prompt, save_name="executer_prompt.txt", per_iteration=True, add_uuid=True
        )

        return prompt

    def parse(self, response: Dict) -> Tuple[str, Optional[str]]:
        """Parse the LLM's response to extract decision and error summary."""

        # Extract content from LLM response
        if isinstance(response, dict) and "content" in response:
            content = response["content"]
        elif isinstance(response, str):
            content = response
        else:
            logger.warning("Unexpected response format from LLM")
            return "FIX", "Parser error"

        # Parse the decision
        decision = "FIX"  # Default to FIX if parsing fails
        if "DECISION:" in content:
            decision_line = [line for line in content.split("\n") if "DECISION:" in line]
            if decision_line:
                decision_text = decision_line[0].split("DECISION:")[1].strip()
                if "FINISH" in decision_text.upper():
                    decision = "FINISH"
                elif "FIX" in decision_text.upper():
                    decision = "FIX"

        # Parse the error summary
        error_summary = None
        if "ERROR_SUMMARY:" in content:
            error_summary_parts = content.split("ERROR_SUMMARY:")[1].strip()
            error_summary = error_summary_parts.split("\n\n")[0].strip()
            if error_summary.lower() == "none" or not error_summary:
                error_summary = None

        self.manager.save_and_log_states(
            content=response, save_name="executer_response.txt", per_iteration=True, add_uuid=True
        )
        self.manager.save_and_log_states(content=decision, save_name="decision.txt", per_iteration=True, add_uuid=True)
        self.manager.save_and_log_states(
            content=error_summary, save_name="error_summary.txt", per_iteration=True, add_uuid=True
        )

        return decision, error_summary
