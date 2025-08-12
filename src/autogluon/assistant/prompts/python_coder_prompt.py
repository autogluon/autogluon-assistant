"""
Python code generation prompt.

This module provides the PythonCoderPrompt class for generating Python code
based on task description, data structure, and other context.
"""

import logging
from typing import Dict, Optional, Tuple

from ..utils import get_cpu_count, get_gpu_count
from .base_prompt import BasePrompt
from .utils import extract_code

logger = logging.getLogger(__name__)


class PythonCoderPrompt(BasePrompt):
    """Handles prompts for Python code generation"""

    def default_template(self) -> str:
        return """
As an AutoML Agent, you will be given a folder containing data and description files. Please generate Python code using {selected_tool} to train a predictor and make predictions on test data. Follow these specifications:

ONLY save files to the working directory: {output_folder}.

1. Data preprocessing:
   - Remove training data samples without valid labels (drop NA values from training dataset ONLY, NOT from test dataset) unless explicitly instructed otherwise.
   - Remove the unneccesary index column (if applicable)

2. Model training:
   - Use {selected_tool} with appropriate parameters for the task
   - If a model is trained, save it in a folder with random timestamp within {output_folder}

3. Prediction:
   - Make predictions on the test data. Always preserve and use the ORIGINAL INDICES from the test data to maintain exact row correspondence - DO NOT generate new indices or rely on assumed ordering.
   - Save the predicted results to {output_folder}, result file name should be "results", the format and extension should be same as the test data file
   - Output column names must exactly match those in the training or sample submission files without adding "predicted_" prefixes or creating any new columns.

4. Documentation:
   - Add a brief docstring at the beginning of the script explaining its purpose
   - Include additional installation steps with comments at the beginning of the script
   - Include comments explaining any complex operations or design decisions

5. Others:
   - To avoid DDP errors, wrap the code in: if __name__ == "__main__":
   - Ensure errors are propagated up and not silently caught - do not use try/except blocks unless you explicitly re-raise the exception.

{validation_prompt}

{tool_prompt}

{best_code_prompt}

Please provide the complete Python script that accomplishes these tasks, ensuring it's ready to run given the appropriate data inputs.

### Task Description
{task_description}

### Data Structure
{data_prompt}

### User Instruction
{user_input}

### Previous Errors
{error_prompt}

### Tutorials for Reference
{tutorial_prompt}
"""

    def get_format_instruction(self) -> str:
        """Get the format instruction to append to the prompt."""
        return "Please format your response with the code in a ```python``` code block to make it easily extractable."

    def build(self) -> str:
        """Build a prompt for the LLM to generate Python code."""
        assert self.manager.time_step >= 0, "run manager.step(user_input) before retrieving the prompt"

        # Truncate outputs if they exceed max length
        if self.manager.user_input:
            user_input = self._truncate_output_end(self.manager.user_input, self.manager.config.max_user_input_length)
        else:
            user_input = "N/A"

        # Generate best code prompt and validation prompt
        best_code_prompt = self._generate_best_code_prompt()
        validation_prompt = self._generate_validation_prompt()

        # Render the prompt using the variable provider with additional variables
        additional_vars = {
            "user_input": user_input,  # Override with truncated version
            "best_code_prompt": best_code_prompt,  # Dynamically generated
            "validation_prompt": validation_prompt,  # Dynamically generated
        }
        
        prompt = self.render(additional_vars)

        # TODO: Remove hardcoding. And add this safeguard for other prompts.
        if len(prompt) > 80000:
            logger.warning(f"Coder's prompt too long: {len(prompt)}. Truncated.")
            self.manager.save_and_log_states(
                content=prompt,
                save_name="python_coder_prompt_before_truncation.txt",
                per_iteration=True,
                add_uuid=False,
            )
            prompt = self._truncate_output_end(
                output=prompt,
                max_length=80000,
            )

        self.manager.save_and_log_states(
            content=prompt, save_name="python_coder_prompt.txt", per_iteration=True, add_uuid=False
        )

        return prompt

    def _generate_validation_prompt(self) -> str:
        """Generate the validation section of the prompt."""
        if self.manager.config.continuous_improvement:
            return """6. Validation:
   - If no validation data is given, hold out a validation dataset (10 percent of the data) at the start, train only on the remaining data.
   - At the end compute and print the final evaluation metric score on the validation set.
   - Use a try-except block for the validation step - if validation fails, it's acceptable to continue.
"""
        else:
            return ""

    def _generate_system_resources_prompt(self) -> str:
        """Generate information about available system resources."""
        return f"""### System Resources
Available CPUs: {get_cpu_count()}
Available GPUs: {get_gpu_count()}
Please optimize your code to efficiently utilize the available hardware resources. 
"""

    def _generate_best_code_prompt(self) -> str:
        """Generate prompt section about best/successful previous code."""
        if self.manager.time_step == 0:
            return ""  # No previous code on first iteration

        best_code_prompt = []

        # Check if we have a best step with validation score
        if self.manager.best_step >= 0 and self.manager.best_step < self.manager.time_step:
            best_code = self.manager.python_codes[self.manager.best_step]
            best_score = self.manager.val_scores[self.manager.best_step]

            best_code_prompt.append("### Previous Best Code")
            best_code_prompt.append(
                f"The following code achieved the best validation score so far ({best_score:.4f}):"
            )
            best_code_prompt.append("```python")
            best_code_prompt.append(best_code)
            best_code_prompt.append("```")
            best_code_prompt.append("")
            best_code_prompt.append(
                "Please prioritize model architecture improvements and training optimization to enhance performance. Feature engineering may also be applied but with lower priority."
            )
            if self.manager.config.optimize_system_resources:
                best_code_prompt.append(self._generate_system_resources_prompt())
        # Check if we have a last successful step (different from best step)
        elif self.manager.last_successful_step >= 0 and self.manager.last_successful_step < self.manager.time_step:
            successful_code = self.manager.python_codes[self.manager.last_successful_step]

            best_code_prompt.append("### Previous Successful Code")
            best_code_prompt.append("The following code executed successfully:")
            best_code_prompt.append("```python")
            best_code_prompt.append(successful_code)
            best_code_prompt.append("```")
            best_code_prompt.append("")
            best_code_prompt.append(
                "Please prioritize model architecture improvements and training optimization to enhance performance. Feature engineering may also be applied but with lower priority."
            )
            if self.manager.config.optimize_system_resources:
                best_code_prompt.append(self._generate_system_resources_prompt())
        # Do nothing if there's no successful code
        else:
            best_code_prompt = []

        return "\n".join(best_code_prompt)

    def parse(self, response: Dict) -> Tuple[str, Optional[str]]:
        """Parse the LLM's response to generated python code"""

        python_code = extract_code(response=response, language="python")

        self.manager.save_and_log_states(
            content=response, save_name="python_coder_response.txt", per_iteration=True, add_uuid=False
        )
        self.manager.save_and_log_states(
            content=python_code, save_name="python_code.py", per_iteration=True, add_uuid=False
        )

        return python_code
