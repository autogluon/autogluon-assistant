import logging
from typing import Dict, Optional, Tuple

from .base_prompt import BasePrompt
from .utils import extract_code

logger = logging.getLogger(__name__)


class PythonCoderPrompt(BasePrompt):
    """Handles prompts for code execution evaluation"""

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
   - Make predictions on the test data. Always preserve and use the original indexing column from the test data to maintain exact row correspondence - do not generate new indices or rely on assumed ordering.
   - Save the predicted results to {output_folder}, result file name should be "results", the format and extension should be same as the test data file
   - Output column names must exactly match those in the training or sample submission files without adding "predicted_" prefixes or creating any new columns.

4. Documentation:
   - Add a brief docstring at the beginning of the script explaining its purpose
   - Include additional installation steps with comments at the beginning of the script
   - Include comments explaining any complex operations or design decisions

5. Others:
   - To avoid DDP errors, wrap the code in: if __name__ == "__main__":
   - Ensure errors are propagated up and not silently caught - do not use try/except blocks unless you explicitly re-raise the exception.

{tool_prompt}

Please provide the complete Python script that accomplishes these tasks, ensuring it's ready to run given the appropriate data inputs.

### Task Description
{task_description}

### Data Structure
{data_prompt}

### User Instruction
{user_prompt}

### Previous Errors
{error_prompt}

### Tutorials for Reference
{tutorial_prompt}
"""

    def build(self) -> str:
        """Build a prompt for the LLM to evaluate execution logs."""

        assert self.manager.time_step >= 0, "run manager.step(user_input) before retriving the prompt"

        # Truncate outputs if they exceed max length
        if self.manager.user_input:
            user_prompt = self._truncate_output_end(self.manager.user_input, self.manager.config.max_user_input_length)
        else:
            user_prompt = "N/A"

        # Format the prompt using the template
        prompt = self.template.format(
            output_folder=self.manager.output_folder,
            selected_tool=self.manager.selected_tool,
            tool_prompt=self.manager.tool_prompt,
            task_description=self.manager.task_description,  # TODO: add task_description in manager
            data_prompt=self.manager.data_prompt,
            user_prompt=user_prompt,
            error_prompt=self.manager.all_previous_error_prompts,
            tutorial_prompt=self.manager.tutorial_prompt,
        )

        # Add format instruction if configured
        if self.llm_config.add_coding_format_instruction:
            format_instruction = (
                "Please format your response with the code in a ```python``` code block to make it easily extractable."
            )
            prompt = f"{prompt}\n\n{format_instruction}"

        self.manager.save_and_log_states(
            content=prompt, save_name="python_coder_prompt.txt", per_iteration=True, add_uuid=False
        )

        return prompt

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
