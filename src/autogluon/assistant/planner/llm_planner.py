import logging
from datetime import datetime
from typing import Dict, Optional, Tuple

from omegaconf import DictConfig

from ..llm import ChatLLMFactory

logger = logging.getLogger(__name__)


class LLMPlanner:
    def __init__(self, llm_config: DictConfig):
        """Initialize with LLM configuration.
        Args:
            llm_config: Configuration for the LLM model
        """
        self.llm_config = llm_config
        self.multi_turn = llm_config.multi_turn
        if self.multi_turn:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            self.llm = ChatLLMFactory.get_chat_model(llm_config, session_name=f"multi_round_coder_{timestamp}")

    def __call__(
        self,
        stdout: str,
        stderr: str,
        python_code: str,
        task_prompt: str,
        data_prompt: str,
    ) -> Tuple[str, Optional[str], str]:
        """
        Evaluate execution logs to determine next steps.

        Args:
            stdout: Standard output from code execution
            stderr: Standard error from code execution
            python_code: The Python code that was executed
            task_prompt: The original task prompt
            data_prompt: The data description prompt

        Returns:
            Tuple containing:
                - decision: "FINISH" if execution was successful, "FIX" if issues need to be fixed
                - error_summary: Summary of errors if any (None if no errors)
                - prompt: The prompt used for evaluation
        """
        # Create a new LLM instance for each call if not using multi-turn
        if not self.multi_turn:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            self.llm = ChatLLMFactory.get_chat_model(self.llm_config, session_name=f"single_round_planner_{timestamp}")

        # Build prompt for evaluating execution results
        prompt = self._build_evaluation_prompt(
            stdout=stdout,
            stderr=stderr,
            python_code=python_code,
            task_prompt=task_prompt,
            data_prompt=data_prompt,
        )

        # Query the LLM
        response = self.llm.assistant_chat(prompt)

        # Parse the LLM response to extract decision and error summary
        decision, error_summary = self._parse_evaluation_response(response)

        # Log the decision and error summary
        logger.info(f"Planner decision: {decision}")
        if error_summary:
            logger.info(f"Error summary: {error_summary}")

        return decision, error_summary, prompt

    def _build_evaluation_prompt(
        self,
        stdout: str,
        stderr: str,
        python_code: str,
        task_prompt: str,
        data_prompt: str,
    ) -> str:
        """
        Build a prompt for the LLM to evaluate execution logs.

        Args:
            stdout: Standard output from code execution
            stderr: Standard error from code execution
            python_code: The Python code that was executed
            task_prompt: The original task prompt
            data_prompt: The data description prompt

        Returns:
            Prompt string for LLM evaluation
        """

        # Truncate stdout if it exceeds max length (from bottom to top)
        if len(stdout) > self.llm_config.max_stdout_length:
            truncated_text = f"\n[...TRUNCATED ({len(stdout) - self.llm_config.max_stdout_length} characters)...]\n"
            stdout = truncated_text + stdout[-self.llm_config.max_stdout_length :]

        # Truncate stderr if it exceeds max length (from bottom to top)
        if len(stderr) > self.llm_config.max_stderr_length:
            truncated_text = f"\n[...TRUNCATED ({len(stderr) - self.llm_config.max_stderr_length} characters)...]\n"
            stderr = truncated_text + stderr[-self.llm_config.max_stderr_length :]

        prompt = f"""You are an expert code evaluator. Analyze the execution results of the following Python code and determine if the execution was successful or if issues need to be fixed.

{task_prompt}{data_prompt}

## Python Code
```python
{python_code}
```

## Execution Results
### Standard Output (stdout)
```
{stdout or "No standard output"}
```

### Standard Error (stderr)
```
{stderr or "No standard error"}
```

Evaluate the execution results and decide on one of the following actions:
1. FINISH - If the execution was completely successful and met all requirements.
2. FIX - If there were errors, issues, or performance problems that need to be addressed.

Provide your decision in the following format:
DECISION: [FINISH or FIX]
ERROR_SUMMARY: [Brief summary of errors if any, or "None" if no errors]

The error summary should be brief but informative enough for another agent to understand what needs to be fixed.

Even if the code executed without throwing errors, it might still have issues with logic or not meet all requirements."""

        return prompt

    def _parse_evaluation_response(self, response: Dict) -> Tuple[str, Optional[str]]:
        """
        Parse the LLM's response to extract decision and error summary.

        Args:
            response: LLM response dictionary

        Returns:
            Tuple of (decision, error_summary)
        """
        # Extract content from LLM response
        if isinstance(response, dict) and "content" in response:
            content = response["content"]
        elif isinstance(response, str):
            content = response
        else:
            # Default values if response format is unexpected
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

        return decision, error_summary
