import logging
from pathlib import Path
from typing import List

from .data_prompt import generate_data_prompt_with_llm
from .error_prompt import generate_error_prompt
from .execution_prompt import generate_execution_prompt
from .task_prompt import generate_task_prompt
from .tutorial_prompt import generate_tutorial_prompt
from .user_prompt import generate_user_prompt


# Create a logger
logger = logging.getLogger(__name__)


class PromptGenerator:
    def __init__(
        self,
        input_data_folder: str,
        output_folder: str,
        config: str,
    ):
        """Initialize PromptGenerator with required paths and config from YAML file.

        Args:
            input_data_folder: Path to input data directory
            output_folder: Path to output directory
            config_path: Path to YAML configuration file
        """
        # Store required paths
        self.input_data_folder = input_data_folder
        self.output_folder = output_folder

        # Validate paths
        for path, name in [(input_data_folder, "input_data_folder")]:
            if not Path(path).exists():
                raise FileNotFoundError(f"{name} not found: {path}")

        # Create output folder if it doesn't exist
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        # Create prompts folder
        self.prompts_folder = Path(output_folder) / "prompts"
        self.prompts_folder.mkdir(parents=True, exist_ok=True)

        self.config = config
        self.coder_multi_turn = config.coder.multi_turn

        # Initialize prompts
        initial_prompts = self.generate_initial_prompts()
        self.task_prompt = initial_prompts["task_prompt"]
        self.data_prompt = initial_prompts["data_prompt"]

        # Save initial prompts
        self._save_prompt("task_prompt", self.task_prompt)
        self._save_prompt("data_prompt", self.data_prompt)

        self.user_inputs: List[str] = []
        self.error_messages: List[str] = []
        self.error_prompts: List[str] = []
        self.python_codes: List[str] = []
        self.bash_scripts: List[str] = []
        self.tutorial_prompts: List[str] = []

        self.time_step = -1

    def _save_prompt(self, prompt_type: str, content: str, step: int = None):
        """Save a prompt to the prompts folder.

        Args:
            prompt_type: Type of the prompt (e.g., 'task', 'data', 'user')
            content: The prompt content to save
            step: Optional step number for iterative prompts
        """
        if step is not None:
            filename = f"{prompt_type}_step_{step}.txt"
        else:
            filename = f"{prompt_type}.txt"

        file_path = self.prompts_folder / filename
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Saved {prompt_type} prompt to {file_path}")

    def generate_initial_prompts(self):
        data_prompt = generate_data_prompt_with_llm(
            input_data_folder=self.input_data_folder,
            max_chars_per_file=self.config.max_chars_per_file,
            llm_config=self.config.file_reader,
        )

        task_prompt, self.selected_tool = generate_task_prompt(
            data_prompt=data_prompt,
            output_folder=self.output_folder,
            llm_config=self.config.llm,
        )

        return {"task_prompt": task_prompt, "data_prompt": data_prompt}

    @property
    def user_input(self) -> str:
        assert self.time_step >= 0, "No user input because the prompt generator is not stepped yet."
        assert len(self.user_inputs) == self.time_step + 1, "user input is not updated yet"
        return self.user_inputs[self.time_step]

    @property
    def python_code(self) -> str:
        assert self.time_step >= 0, "No python code because the prompt generator is not stepped yet."
        assert len(self.python_codes) == self.time_step + 1, "python code is not updated yet"
        return self.python_codes[self.time_step]

    @property
    def previous_python_code(self) -> str:
        if self.time_step >= 1:
            return self.python_codes[self.time_step - 1]
        else:
            return ""

    @property
    def bash_script(self) -> str:
        assert self.time_step >= 0, "No bash script because the prompt generator is not stepped yet."
        assert len(self.bash_scripts) == self.time_step + 1, "bash script is not updated yet"
        return self.bash_scripts[self.time_step]

    @property
    def previous_bash_script(self) -> str:
        if self.time_step >= 1:
            return self.bash_scripts[self.time_step - 1]
        else:
            return ""

    @property
    def error_message(self) -> str:
        assert self.time_step >= 0, "No error message because the prompt generator is not stepped yet."
        assert len(self.error_messages) == self.time_step + 1, "error message is not updated yet"
        return self.error_messages[self.time_step]

    @property
    def previous_error_message(self) -> str:
        if self.time_step >= 1:
            return self.error_messages[self.time_step - 1]
        else:
            return ""

    @property
    def error_prompt(self) -> str:
        assert self.time_step >= 0, "No error prompt because the prompt generator is not stepped yet."
        assert len(self.error_prompts) == self.time_step + 1, "error prompt is not updated yet"
        return self.error_prompts[self.time_step]

    @property
    def previous_error_prompt(self) -> str:
        if self.time_step >= 1:
            return self.error_prompts[self.time_step - 1]
        else:
            return ""

    @property
    def tutorial_prompt(self) -> str:
        assert self.time_step >= 0, "No tutorial prompt because the prompt generator is not stepped yet."
        assert len(self.tutorial_prompts) == self.time_step + 1, "tutorial prompt is not updated yet"
        return self.tutorial_prompts[self.time_step]

    @property
    def previous_tutorial_prompt(self) -> str:
        if self.time_step >= 1:
            return self.tutorial_prompts[self.time_step - 1]
        else:
            return ""

    def step(self, user_input=None):
        """Step the prompt generator forward.

        Args:
            user_inputs: Optional user inputs to generate user prompt
            error_message: Optional error message to generate error prompt
        """
        self.time_step += 1

        user_prompt = generate_user_prompt(
            user_input=user_input,
            max_user_input_length=self.config.max_user_input_length,
        )

        # Save user prompt
        if user_input:
            self._save_prompt("user_prompt", user_prompt, self.time_step)

        if self.time_step > 0:
            previous_error_prompt = generate_error_prompt(
                task_prompt=self.task_prompt,
                data_prompt=self.data_prompt,
                user_prompt=user_prompt,
                python_code=self.previous_python_code,
                bash_script=self.previous_bash_script,
                tutorial_prompt=self.previous_tutorial_prompt,
                error_message=self.previous_error_message,
                llm_config=self.config.llm,
                output_folder=self.output_folder,
                max_error_message_length=self.config.max_error_message_length,
                error_summary=self.config.error_summary if hasattr(self.config, "error_summary") else True,
                error_fix=self.config.error_fix if hasattr(self.config, "error_fix") else True,
            )
            assert len(self.error_prompts) == self.time_step - 1
            self.error_prompts.append(previous_error_prompt)

            # Save error prompt
            self._save_prompt("error_prompt", previous_error_prompt, self.time_step - 1)

        tutorial_prompt = generate_tutorial_prompt(
            task_prompt=self.task_prompt,
            data_prompt=self.data_prompt,
            user_prompt=user_prompt,
            error_prompt=self.previous_error_prompt,
            tool_name=self.selected_tool,
            llm_config=self.config.llm,
            output_folder=self.output_folder,
            max_num_tutorials=self.config.max_num_tutorials,
            max_tutorial_length=self.config.max_tutorial_length,
            condense_tutorials=self.config.condense_tutorials,
            use_tutorial_summary=(
                self.config.use_tutorial_summary if hasattr(self.config, "use_tutorial_summary") else True
            ),
        )

        # Save tutorial prompt
        if tutorial_prompt:
            self._save_prompt("tutorial_prompt", tutorial_prompt, self.time_step)

        assert len(self.user_inputs) == self.time_step
        self.user_inputs.append(user_input)

        assert len(self.tutorial_prompts) == self.time_step
        self.tutorial_prompts.append(tutorial_prompt)

    def get_coding_prompt(self) -> str:
        """Get the complete iterative prompt.

        Returns:
            str: The complete prompt combining task, data, user, error and tutorial prompts
        """
        assert self.time_step >= 0, "run PromptGenerator.step(user_input) before get the prompt"

        prompt_parts = []

        # if self.time_step == 0 or not self.coder_multi_turn:
        #   prompt_parts.extend([self.task_prompt, self.data_prompt])
        # else:
        #    prompt_parts.append("Fix the error and return the FULL python script instead of only the correction.")  # TODO: A temp fix to avoid LLM only return code patch
        prompt_parts.extend(
            [self.task_prompt, self.data_prompt]
        )  # TODO: Performance Degrade without providing init prompt

        if self.user_input:
            user_prompt = generate_user_prompt(
                user_input=self.user_input,
                max_user_input_length=self.config.max_user_input_length,
            )
            prompt_parts.append(user_prompt)

        if self.time_step == 0 or not self.coder_multi_turn:
            for error_prompt in self.error_prompts:
                prompt_parts.append(error_prompt)
        else:
            prompt_parts.append(self.previous_error_prompt)

        if self.tutorial_prompt:
            prompt_parts.append(self.tutorial_prompt)

        complete_prompt = "\n\n".join(prompt_parts)

        # Save the complete coding prompt
        self._save_prompt("complete_coding_prompt", complete_prompt, self.time_step)

        return complete_prompt

    def get_execution_prompt(self, python_file_path) -> str:
        install_packages = "machine learning" in self.selected_tool
        self.execution_prompt = generate_execution_prompt(
            output_folder=self.output_folder,
            python_file_path=python_file_path,
            create_venv=self.config.create_venv,
            install_packages=install_packages,
            previous_bash=self.previous_bash_script,
            previous_python=self.previous_python_code,
            current_python=self.python_code,
            error_message=self.previous_error_message,
            max_error_message_length=self.config.max_error_message_length,
        )

        # Save the execution prompt
        self._save_prompt("execution_prompt", self.execution_prompt, self.time_step)

        return self.execution_prompt

    def update_python_code(self, python_code: str):
        """Update the current Python code."""
        assert len(self.python_codes) == self.time_step
        self.python_codes.append(python_code)

    def update_bash_script(self, bash_script: str):
        """Update the current bash script."""
        assert len(self.bash_scripts) == self.time_step
        self.bash_scripts.append(bash_script)

    def update_error_message(self, error_message: str):
        """Update the current error message."""
        assert len(self.error_messages) == self.time_step
        self.error_messages.append(error_message)
