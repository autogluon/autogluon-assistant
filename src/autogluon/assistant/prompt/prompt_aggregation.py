import logging
from pathlib import Path
from typing import List

from ..agents import DataPerceptionAgent, ErrorAnalyzerAgent
from ..tools_registry import registry
from .task_prompt import generate_task_prompt
from .tutorial_prompt import generate_tutorial_prompt
from .user_prompt import generate_user_prompt

# Basic configuration
logging.basicConfig(level=logging.INFO)

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
        self.python_file_paths: List[str] = []
        self.bash_scripts: List[str] = []
        self.tutorial_prompts: List[str] = []

        self.error_analyzer = ErrorAnalyzerAgent(
            config=self.config, 
            llm_config=self.config.error_analyzer, 
            prompt_template=None,  # TODO: Add prompt_template to argument
        )

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
        self.dp_agent = DataPerceptionAgent(
            config=self.config,
            input_data_folder=self.input_data_folder,
            reader_llm_config=self.config.reader,
            reader_prompt_template=None,  # TODO: add it to argument
        )

        data_prompt = self.dp_agent()

        task_prompt, self.selected_tool, self.task_description = generate_task_prompt(
            data_prompt=data_prompt,
            output_folder=self.output_folder,
            llm_config=self.config.llm,
        )

        # Get tool-specific template and requirements if they exist
        tool_info = registry.get_tool(self.selected_tool)
        if not tool_info:
            raise ValueError(f"Tool {self.selected_tool} not found in registry")
        # Get tool-specific prompt
        self.tool_prompt = tool_info.get("prompt_template", "")
        if isinstance(self.tool_prompt, list):
            self.tool_prompt = "\n".join(self.tool_prompt)

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
    def python_file_path(self) -> str:
        assert self.time_step >= 0, "No python file path because the prompt generator is not stepped yet."
        assert len(self.python_file_paths) == self.time_step + 1, "python file path is not updated yet"
        return self.python_file_paths[self.time_step]

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
        
        assert len(self.user_inputs) == self.time_step
        self.user_inputs.append(user_input)

        if self.time_step > 0:
            previous_error_prompt = self.error_analyzer(self)

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

    def update_python_code(self, python_code: str, python_file_path: str):
        """Update the current Python code."""
        assert len(self.python_codes) == self.time_step
        assert len(self.python_file_paths) == self.time_step

        self.python_codes.append(python_code)
        self.python_file_paths.append(python_file_path)

    def update_bash_script(self, bash_script: str):
        """Update the current bash script."""
        assert len(self.bash_scripts) == self.time_step
        self.bash_scripts.append(bash_script)

    def update_error_message(self, error_message: str):
        """Update the current error message."""
        assert len(self.error_messages) == self.time_step
        self.error_messages.append(error_message)
