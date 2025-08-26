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
{execution_task}

### Data Structure
{execution_data}

### Python Code
{code_to_analyze}

## Execution Results
### Standard Output (stdout)

{stdout_truncate_mid_8192}

### Standard Error (stderr)

{stderr_truncate_mid_8192}

Evaluate the execution results and decide on one of the following actions:
1. SUCCESS - If the execution was completely successful and met all requirements.
2. FIX - If there were errors, issues, or performance problems that need to be addressed.

Provide your decision in the following format:
DECISION: [SUCCESS or FIX]
ERROR_SUMMARY: [Brief summary of errors if any, or "None" if no errors]
VALIDATION_SCORE: [If there is a validation score for the solution, provide it as a number, otherwise "None"]

The error summary should be brief but informative enough for another agent to understand what needs to be fixed.
Even if the code executed without throwing errors, it might still have issues with logic or not meet all requirements.

For validation scores:
- If there is a validation score present in the execution results, extract it
- Convert the score to ensure higher values indicate better performance (multiply "lower is better" metrics like RMSE, MAE, or loss by -1)
- Return the converted score that follows the "higher is better" convention"""

    def _build(self, stdout: str, stderr: str, code_to_analyze: str, execution_task: str, execution_data: str) -> str:
        """Build a prompt for the LLM to evaluate execution logs."""
        self.manager.save_and_log_states(content=stdout, save_name="stdout.txt", per_iteration=True, add_uuid=True)
        self.manager.save_and_log_states(content=stderr, save_name="stderr.txt", per_iteration=True, add_uuid=True)

        # Save original stdout and stderr
        self.manager.save_and_log_states(
            content=stdout, save_name="stdout.orig.txt", per_iteration=True, add_uuid=True
        )
        self.manager.save_and_log_states(
            content=stderr, save_name="stderr.orig.txt", per_iteration=True, add_uuid=True
        )

        # Render the prompt using the variable provider with additional variables
        additional_vars = {
            "execution_task": execution_task,
            "execution_data": execution_data,
            "code_to_analyze": code_to_analyze,
            "stdout": stdout or "No standard output",
            "stderr": stderr or "No standard error",
        }

        prompt = self.render(additional_vars)

        self.manager.save_and_log_states(
            content=prompt, save_name="executer_prompt.txt", per_iteration=True, add_uuid=True
        )

        return prompt

    def parse(self, response: Dict) -> Tuple[str, Optional[str], Optional[float]]:
        """Parse the LLM's response to extract decision, error summary, and validation score."""

        # Extract content from LLM response
        if isinstance(response, dict) and "content" in response:
            content = response["content"]
        elif isinstance(response, str):
            content = response
        else:
            logger.warning("Unexpected response format from LLM")
            return "FIX", "Parser error", None

        # Parse the decision
        decision = "FIX"  # Default to FIX if parsing fails
        if "DECISION:" in content:
            decision_line = [line for line in content.split("\n") if "DECISION:" in line]
            if decision_line:
                decision_text = decision_line[0].split("DECISION:")[1].strip()
                if "SUCCESS" in decision_text.upper():
                    decision = "SUCCESS"
                elif "FIX" in decision_text.upper():
                    decision = "FIX"

        # Parse the error summary
        error_summary = None
        if "ERROR_SUMMARY:" in content:
            error_summary_parts = content.split("ERROR_SUMMARY:")[1].strip()
            error_summary = error_summary_parts.split("\n")[0].strip()
            if error_summary.lower() == "none" or not error_summary:
                error_summary = None

        # Parse the validation score
        validation_score = None
        if "VALIDATION_SCORE:" in content:
            validation_score_parts = content.split("VALIDATION_SCORE:")[1].strip()
            validation_score_text = validation_score_parts.split("\n")[0].strip()
            if validation_score_text.lower() != "none" and validation_score_text:
                try:
                    validation_score = float(validation_score_text)
                except ValueError:
                    logger.warning(f"Could not parse validation score: {validation_score_text}")
                    validation_score = None
        # The Validation score is only meaningful if this is a success run
        if decision != "SUCCESS":
            validation_score = None

        self.manager.save_and_log_states(
            content=response, save_name="executer_response.txt", per_iteration=True, add_uuid=True
        )
        self.manager.save_and_log_states(content=decision, save_name="decision.txt", per_iteration=True, add_uuid=True)
        self.manager.save_and_log_states(
            content=error_summary, save_name="error_summary.txt", per_iteration=True, add_uuid=True
        )
        self.manager.save_and_log_states(
            content=str(validation_score), save_name="validation_score.txt", per_iteration=True, add_uuid=True
        )

        return decision, error_summary, validation_score
