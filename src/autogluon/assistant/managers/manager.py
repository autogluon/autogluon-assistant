import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import List, Optional

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
from ..constants import ENV_FOLDER_NAME
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
        self.best_step = -1
        self.last_successful_step = -1
        self.best_step_saved = -1

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
        self.val_scores: List[Optional[float]] = []

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

        # TODO: remove the hard code for "create_venv" (add in tool registry if need installation)
        if self.selected_tool.lower() in ["machine learning", "huggingface", "fairseq"]:
            self.config.create_venv = True

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
    def common_env_file(self) -> str:
        return registry.registry_path / "_common" / "requirements.txt"

    @property
    def selected_tool_env_file(self) -> str:
        tool_path = registry.get_tool(self.selected_tool)["path"]
        return registry.registry_path / tool_path / "requirements.txt"

    @property
    def iteration_folder(self) -> str:
        if self.time_step >= 0:
            iter_folder = os.path.join(self.output_folder, f"generation_iter_{self.time_step}")
        else:
            iter_folder = os.path.join(self.output_folder, "initialization")
        os.makedirs(iter_folder, exist_ok=True)
        return iter_folder

    @property
    def per_iteration_output_folder(self) -> str:
        iter_output_folder = os.path.join(self.iteration_folder, "output")
        os.makedirs(iter_output_folder, exist_ok=True)
        return iter_output_folder

    @property
    def validation_score(self) -> Optional[float]:
        """Get the current validation score."""
        assert self.time_step >= 0, "No validation score because the prompt generator is not stepped yet."
        assert len(self.val_scores) == self.time_step + 1, "validation score is not updated yet"
        return self.val_scores[self.time_step]

    @property
    def best_validation_score(self) -> Optional[float]:
        """Get the best validation score found so far."""
        if self.best_step >= 0 and self.best_step < len(self.val_scores):
            return self.val_scores[self.best_step]
        return None

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
                    f"[bold green]Previous iteration info is stored in:[/bold green] {os.path.join(self.output_folder, f'iteration_{self.time_step - 1}')}"
                )
            else:
                logger.brief(
                    f"[bold green]Initialization info is stored in:[/bold green] {os.path.join(self.output_folder, 'initialization')}"
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
        planner_decision, planner_error_summary, validation_score, planner_prompt, stderr, stdout = self.executer(
            code_to_execute=self.bash_script,
            code_to_analyze=self.python_code,
            task_description=self.task_description,
            data_prompt=self.data_prompt,
        )

        self.save_and_log_states(stderr, "stderr", per_iteration=True, add_uuid=False)
        self.save_and_log_states(stdout, "stdout", per_iteration=True, add_uuid=False)

        # Track validation scores and update best step
        assert len(self.val_scores) == self.time_step
        self.val_scores.append(validation_score)

        # Update best step if we have a better validation score (higher is better)
        if validation_score is not None:
            if self.best_step == -1 or validation_score > self.val_scores[self.best_step]:
                self.best_step = self.time_step
                logger.brief(
                    f"[bold green]New best validation score: {validation_score:.4f} at step {self.time_step}[/bold green]"
                )
            else:
                logger.brief(
                    f"[bold yellow]Current validation score: {validation_score:.4f} (best: {self.val_scores[self.best_step]:.4f} at step {self.best_step})[/bold yellow]"
                )

        # Save validation score information
        self.save_and_log_states(
            content=f"Step: {self.time_step}\nValidation Score: {validation_score}\nBest Step: {self.best_step}\nBest Score: {self.best_validation_score}",
            save_name="validation_score.txt",
            per_iteration=True,
            add_uuid=False,
        )

        if planner_decision == "FIX":
            logger.brief(f"[bold red]Code generation failed in iteration[/bold red] {self.time_step}!")
            # Add suggestions to the error message to guide next iteration
            error_message = f"stderr: {stderr}\n\n" if stderr else ""
            error_message += (
                f"Error summary from planner (the error can appear in stdout if it's catched): {planner_error_summary}"
            )
            self.update_error_message(error_message=error_message)
            self.remove_env_folder(self.iteration_folder)
            return False
        elif planner_decision == "SUCCESS":
            self.last_successful_step = self.time_step
            logger.brief(f"[bold green]Code generation successful at iteration[/bold green] {self.time_step}")
            if validation_score is not None:
                logger.brief(f"[bold green]Final validation score: {validation_score:.4f}[/bold green]")
            if self.best_step >= 0:
                logger.brief(
                    f"[bold green]Best validation score achieved: {self.best_validation_score:.4f} at step {self.best_step}[/bold green]"
                )
            self.update_error_message(error_message="")
            return True
        else:
            logger.warning(f"###INVALID Planner Output: {planner_decision}###")
            self.update_error_message(error_message="")
            self.remove_env_folder(self.iteration_folder)
            return False

    def update_error_message(self, error_message: str):
        """Update the current error message."""
        assert len(self.error_messages) == self.time_step
        self.error_messages.append(error_message)

    def get_validation_score_summary(self) -> str:
        """Get a summary of all validation scores."""
        if not self.val_scores:
            return "No validation scores available."

        summary = ["Validation Score Summary:"]
        for i, score in enumerate(self.val_scores):
            marker = " (BEST)" if i == self.best_step else ""
            summary.append(f"Step {i}: {score if score is not None else 'N/A'}{marker}")

        if self.best_step >= 0:
            summary.append(f"\nBest score: {self.best_validation_score:.4f} at step {self.best_step}")

        return "\n".join(summary)

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

    def create_best_run_copy(self):
        """Create a 'best_run' folder that copies the best step folder.

        If no best step is available, uses the last successful step.
        If neither is available, logs a warning and does nothing.
        """

        # Determine which step to copy
        target_step = None
        copy_reason = ""

        if self.best_step >= 0:
            target_step = self.best_step
            copy_reason = f"best validation score ({self.best_validation_score:.4f})"
        elif self.last_successful_step >= 0:
            target_step = self.last_successful_step
            copy_reason = "last successful execution"
        else:
            logger.warning("No best step or successful step found. Cannot create best_run copy.")
            return

        if target_step == self.best_step_saved:
            logger.info(f"Skipping the saving process as step {target_step} has already been saved as best run.")

        # Create paths
        source_folder = os.path.join(self.output_folder, f"generation_iter_{target_step}")
        best_run_folder = os.path.join(self.output_folder, "best_run")

        # Verify source folder exists
        if not os.path.exists(source_folder):
            logger.warning(f"Source folder does not exist: {source_folder}")
            return

        # Check if source folder has an 'output' subdirectory
        source_output_folder = os.path.join(source_folder, "output")
        if not os.path.exists(source_output_folder):
            logger.warning(f"Source output folder does not exist: {source_output_folder}")
            return

        # Remove existing best_run folder if it exists
        if os.path.exists(best_run_folder):
            try:
                shutil.rmtree(best_run_folder)
                logger.info("Removed existing best_run folder")
            except Exception as e:
                logger.error(f"Failed to remove existing best_run folder: {e}")
                return

        try:
            # Copy all files from source_output_folder to self.output_folder
            for item in os.listdir(source_output_folder):
                source_item = os.path.join(source_output_folder, item)
                dest_item = os.path.join(self.output_folder, item)

                if os.path.isfile(source_item):
                    shutil.copy2(source_item, dest_item)
                elif os.path.isdir(source_item):
                    shutil.copytree(source_item, dest_item, dirs_exist_ok=True)

            # Copy the entire source folder to best_run folder
            shutil.copytree(source_folder, best_run_folder)

            # Remove the env folder from source folder to save space
            self.remove_env_folder(source_folder)

            logger.brief(
                f"[bold green]Created best_run folder (copied from step {target_step} - {copy_reason})[/bold green]"
            )

            # Save summary information in the best_run folder
            summary_content = [
                "Best Run Summary",
                "================",
                f"Copied from: generation_iter_{target_step}",
                f"Reason: {copy_reason}",
                f"Copy created at: {os.path.basename(best_run_folder)}",
                "",
                self.get_validation_score_summary(),
            ]

            # Save summary in both the main output folder and the best_run folder
            summary_text = "\n".join(summary_content)

            self.save_and_log_states(
                content=summary_text, save_name="best_run_summary.txt", per_iteration=False, add_uuid=False
            )

            self.best_step_saved = target_step

        except Exception as e:
            logger.error(f"Failed to copy folder: {e}")
            return

    def remove_env_folder(self, iter_folder):
        if not self.config.cleanup_unused_env:
            return
        try:
            env_folder = os.path.join(iter_folder, ENV_FOLDER_NAME)
            shutil.rmtree(env_folder)
            logger.info(f"Removed unused env folder {env_folder}")
        except Exception as e:
            logger.error(f"Failed to remove env folder {env_folder}: {e}")

    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, "retriever"):
            self.retriever.cleanup()

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
