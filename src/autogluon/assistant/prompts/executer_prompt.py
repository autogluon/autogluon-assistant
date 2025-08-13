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
1. SUCCESS - Final output is correct, regardless of the approach.
2. RESTART - Final output has errors or performance problems due to wrong or incomplete task perception, i.e. in Task Descriptions, Data Structure sections (e.g. error reading data files), or selecting the wrong ML library.
3. FIX - Final output has errors or performance problems, but the task perception is good.
Only choose FIX or RESTART if the final result contains actual errors or performance problems.

Provide your decision in the following format:
DECISION: [SUCCESS, FIX, or RESTART]
ERROR_SUMMARY: [Brief summary of errors if any, or "None" if no errors]
VALIDATION_SCORE: [If there is a validation score for the solution, provide it as a number, otherwise "None"]

The error summary should be brief but informative enough for another agent to understand what needs to be fixed.
Even if the code executed without throwing errors, it might still have issues with logic or not meet all requirements.

For RESTART decisions, the ERROR_SUMMARY will be appended to the initial instruction provided to agents after restart, so it should clearly describe the initialization problem that caused the failure precisely and concisely.

For validation scores:
- If there is a validation score present in the execution results, extract it
- Convert the score to ensure higher values indicate better performance (multiply "lower is better" metrics like RMSE, MAE, or loss by -1)
- Return the converted score that follows the "higher is better" convention"""

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
                elif "RESTART" in decision_text.upper():
                    decision = "RESTART"
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
