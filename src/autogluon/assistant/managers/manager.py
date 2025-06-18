import logging
import os
import uuid
from pathlib import Path
from typing import List

from ..agents import (
    CoderAgent,
    DataPerceptionAgent,
    DescriptionFileRetrieverAgent,
    ErrorAnalyzerAgent,
    ExecuterAgent,
    RerankerAgent,
    RetrieverAgent,
    TaskDescriptorAgent,
    ToolSelectorAgent,
)
from ..llm import ChatLLMFactory
from ..tools_registry import registry
from ..utils import get_user_input_webui

# Basic configuration
logging.basicConfig(level=logging.INFO)

# Create a logger
logger = logging.getLogger(__name__)


class Manager:
    def __init__(
        self,
        input_data_folder: str,
        output_folder: str,
        config: str,
    ):
        """Initialize Manager with required paths and config from YAML file.

        Args:
            input_data_folder: Path to input data directory
            output_folder: Path to output directory
            config_path: Path to YAML configuration file
        """
        self.time_step = -1

        # Store required paths
        self.input_data_folder = input_data_folder
        self.output_folder = output_folder

        # Validate paths
        for path, name in [(input_data_folder, "input_data_folder")]:
            if not Path(path).exists():
                raise FileNotFoundError(f"{name} not found: {path}")

        # Create output folder if it doesn't exist
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        self.config = config
        self.coder_multi_turn = config.coder.multi_turn

        self.dp_agent = DataPerceptionAgent(
            config=self.config,
            manager=self,
            input_data_folder=self.input_data_folder,
            reader_llm_config=self.config.reader,
            reader_prompt_template=None,  # TODO: add it to argument
        )

        self.dfr_agent = DescriptionFileRetrieverAgent(
            config=self.config,
            manager=self,
            llm_config=self.config.description_file_retriever,
            prompt_template=None,  # TODO: add it to argument
        )

        self.td_agent = TaskDescriptorAgent(
            config=self.config,
            manager=self,
            llm_config=self.config.task_descriptor,
            prompt_template=None,  # TODO: add it to argument
        )

        self.ts_agent = ToolSelectorAgent(
            config=self.config,
            manager=self,
            llm_config=self.config.tool_selector,
            prompt_template=None,  # TODO: add it to argument
        )

        # Initialize prompts
        self.generate_initial_prompts()

        self.user_inputs: List[str] = []
        self.error_messages: List[str] = []
        self.error_prompts: List[str] = []
        self.python_codes: List[str] = []
        self.python_file_paths: List[str] = []
        self.bash_scripts: List[str] = []
        self.tutorial_retrievals: List[str] = []
        self.tutorial_prompts: List[str] = []

        self.error_analyzer = ErrorAnalyzerAgent(
            config=self.config,
            manager=self,
            llm_config=self.config.error_analyzer,
            prompt_template=None,  # TODO: Add prompt_template to argument
        )

        self.retriever = RetrieverAgent(
            config=self.config,
            manager=self,
            llm_config=self.config.retriever,
            prompt_template=None,  # TODO: Add prompt_template to argument
        )

        self.reranker = RerankerAgent(
            config=self.config,
            manager=self,
            llm_config=self.config.reranker,
            prompt_template=None,  # TODO: Add prompt_template to argument
        )

        self.python_coder = CoderAgent(
            config=self.config,
            manager=self,
            language="python",
            coding_mode="coder",
            llm_config=self.config.coder,
            prompt_template=None,
        )  # TODO: Add prompt_template to argument
        self.bash_coder = CoderAgent(
            config=self.config,
            manager=self,
            language="bash",
            coding_mode="coder",
            llm_config=self.config.coder,
            prompt_template=None,
        )  # TODO: Add prompt_template to argument

        self.executer = ExecuterAgent(
            config=self.config,
            manager=self,
            language="bash",
            timeout=self.config.per_execution_timeout,
            executer_llm_config=self.config.executer,
            executer_prompt_template=None,
        )  # TODO: Add prompt_template to argument

    def generate_initial_prompts(self):
        self.data_prompt = self.dp_agent()

        self.description_files = self.dfr_agent()

        self.task_description = self.td_agent()

        self.selected_tool = self.ts_agent()

        # Get tool-specific template and requirements if they exist
        tool_info = registry.get_tool(self.selected_tool)
        if not tool_info:
            raise ValueError(f"Tool {self.selected_tool} not found in registry")
        # Get tool-specific prompt
        self.tool_prompt = tool_info.get("prompt_template", "")
        if isinstance(self.tool_prompt, list):
            self.tool_prompt = "\n".join(self.tool_prompt)

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
    def all_previous_error_prompts(self) -> str:
        if self.time_step >= 1:
            return "\n\n".join(self.error_prompts[: self.time_step])
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

    @property
    def tutorial_retrieval(self) -> str:
        assert self.time_step >= 0, "No tutorial retrieval because the prompt generator is not stepped yet."
        assert len(self.tutorial_retrievals) == self.time_step + 1, "tutorial retrieval is not updated yet"
        return self.tutorial_retrievals[self.time_step]

    @property
    def previous_tutorial_retrieval(self) -> str:
        if self.time_step >= 1:
            return self.tutorial_retrievals[self.time_step - 1]
        else:
            return ""

    @property
    def iteration_folder(self) -> str:
        if self.time_step >= 0:
            iter_folder = os.path.join(self.output_folder, f"generation_iter_{self.time_step}")
        else:
            iter_folder = os.path.join(self.output_folder, "initialization")
        os.makedirs(iter_folder, exist_ok=True)
        return iter_folder

    def set_initial_user_input(self, need_user_input, initial_user_input):
        self.need_user_input = need_user_input
        self.initial_user_input = initial_user_input

    def step(self):
        """Step the prompt generator forward."""
        self.time_step += 1

        user_input = self.initial_user_input
        # Get per iter user inputs if needed
        if self.need_user_input:
            if self.time_step > 0:
                logger.brief(
                    f"\n[bold green]Previous iteration info is stored in:[/bold green] {os.path.join(self.output_folder, f'iteration_{self.time_step - 1}')}"
                )
            else:
                logger.brief(
                    f"\n[bold green]Initialization info is stored in:[/bold green] {os.path.join(self.output_folder, 'initialization')}"
                )
            if user_input is None:
                user_input = ""
                if os.environ.get("AUTOGLUON_WEBUI", "false").lower() == "true":
                    # If running in WebUI, get user input from stdin
                    user_input += "\n" + get_user_input_webui(
                        f"Enter your inputs for current iteration (iter {self.time_step}) (press Enter to skip): "
                    )
                else:
                    user_input += "\n" + input(
                        f"Enter your inputs for current iteration (iter {self.time_step}) (press Enter to skip): "
                    )

        assert len(self.user_inputs) == self.time_step
        self.user_inputs.append(user_input)

        if self.time_step > 0:
            previous_error_prompt = self.error_analyzer()

            assert len(self.error_prompts) == self.time_step - 1
            self.error_prompts.append(previous_error_prompt)

        retrieved_tutorials = self.retriever()
        assert len(self.tutorial_retrievals) == self.time_step
        self.tutorial_retrievals.append(retrieved_tutorials)

        tutorial_prompt = self.reranker()
        assert len(self.tutorial_prompts) == self.time_step
        self.tutorial_prompts.append(tutorial_prompt)

    def write_code_script(self, script, output_code_file):
        with open(output_code_file, "w") as file:
            file.write(script)

    def update_python_code(self):
        """Update the current Python code."""
        assert len(self.python_codes) == self.time_step
        assert len(self.python_file_paths) == self.time_step

        python_code = self.python_coder()

        python_file_path = os.path.join(self.iteration_folder, "generated_code.py")

        self.write_code_script(python_code, python_file_path)

        self.python_codes.append(python_code)
        self.python_file_paths.append(python_file_path)

    def update_bash_script(self):
        """Update the current bash script."""
        assert len(self.bash_scripts) == self.time_step

        bash_script = self.bash_coder()

        bash_file_path = os.path.join(self.iteration_folder, "execution_script.sh")

        self.write_code_script(bash_script, bash_file_path)

        self.bash_scripts.append(bash_script)

    def execute_code(self):
        planner_decision, planner_error_summary, planner_prompt, stderr, stdout = self.executer(
            code_to_execute=self.bash_script,
            code_to_analyze=self.python_code,
            task_description=self.task_description,
            data_prompt=self.data_prompt,
        )

        self.save_and_log_states(stderr, "stderr", per_iteration=True, add_uuid=False)
        self.save_and_log_states(stdout, "stdout", per_iteration=True, add_uuid=False)

        if planner_decision == "FIX":
            logger.brief(f"[bold red]Code generation failed in iteration[/bold red] {self.time_step}!")
            # Add suggestions to the error message to guide next iteration
            error_message = f"stderr: {stderr}\n\n" if stderr else ""
            error_message += (
                f"Error summary from planner (the error can appear in stdout if it's catched): {planner_error_summary}"
            )
            self.update_error_message(error_message=error_message)
            return False
        elif planner_decision == "FINISH":
            logger.brief(
                f"[bold green]Code generation successful after[/bold green] {self.time_step + 1} [bold green]iterations[/bold green]"
            )
            self.update_error_message(error_message="")
            return True
        else:
            logger.warning(f"###INVALID Planner Output: {planner_decision}###")
            self.update_error_message(error_message="")
            return False

    def update_error_message(self, error_message: str):
        """Update the current error message."""
        assert len(self.error_messages) == self.time_step
        self.error_messages.append(error_message)

    def save_and_log_states(self, content, save_name, per_iteration=False, add_uuid=False):
        if add_uuid:
            # Split filename and extension
            name, ext = os.path.splitext(save_name)
            # Generate 4-digit UUID (using first 4 characters of hex)
            uuid_suffix = str(uuid.uuid4()).replace("-", "")[:4]
            save_name = f"{name}_{uuid_suffix}{ext}"

        if per_iteration:
            states_dir = os.path.join(self.iteration_folder, "states")
        else:
            states_dir = os.path.join(self.output_folder, "states")
        os.makedirs(states_dir, exist_ok=True)
        output_file = os.path.join(states_dir, save_name)

        logger.info(f"Saving {output_file}...")
        with open(output_file, "w") as file:
            if content is not None:
                if isinstance(content, list):
                    # Join list elements with newlines
                    file.write("\n".join(str(item) for item in content))
                else:
                    # Handle as string (original behavior)
                    file.write(content)
            else:
                file.write("<None>")

    def log_agent_start(self, message: str):
        logger.brief(message)

    def log_agent_end(self, message: str):
        logger.brief(message)

    def report_token_usage(self):
        token_usage_path = os.path.join(self.output_folder, "token_usage.json")
        usage = ChatLLMFactory.get_total_token_usage(save_path=token_usage_path)
        total = usage["total"]
        logger.brief(
            f"Total tokens — input: {total['total_input_tokens']}, "
            f"output: {total['total_output_tokens']}, "
            f"sum: {total['total_tokens']}"
        )

        logger.info(f"Full token usage detail:\n{usage}")

    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, "retriever"):
            self.retriever.cleanup()

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
