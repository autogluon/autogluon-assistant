import logging

from rich.progress import (
    Progress,
    TextColumn,
)

from ..prompts import ExecuterPrompt
from ..rich_logging import show_progress_bar
from .base_agent import BaseAgent
from .utils import init_llm

logger = logging.getLogger(__name__)


def execute_code(code, language, timeout):
    """
    Execute code with real-time output streaming and timeout and show a linear timeout progress bar..
    Args:
        code (str): The code to execute (Python code or bash script)
        language (str): The language to execute ("python" or "bash")
        timeout (float): Maximum execution time in seconds before terminating the process.
    Returns:
        tuple: (success: bool, stdout: str, stderr: str)
    """
    import select
    import subprocess
    import time

    try:
        # Set up the command based on language
        if language.lower() == "python":
            cmd = ["python", "-c", code]
        elif language.lower() == "bash":
            cmd = ["bash", "-c", code]
        else:
            return False, "", f"Unsupported language: {language}. Use 'python' or 'bash'."

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        stdout_chunks, stderr_chunks = [], []

        # Set up tracking of both output streams
        streams = [process.stdout, process.stderr]

        # Track start time for timeout
        start_time = time.time()

        with Progress(
            TextColumn(f"[bold cyan]Executing {language}:"),
            TextColumn("[bold green]{task.completed:.1f}s[/bold green] [dim](time limit: {task.total:.0f}s)[/dim]"),
            refresh_per_second=2,
            transient=False,
            disable=not show_progress_bar(),
        ) as progress_context:

            task = progress_context.add_task("", total=timeout)

            while streams:
                # Calculate remaining time
                elapsed_time = time.time() - start_time
                progress_context.update(task, completed=elapsed_time)
                remaining_time = max(0, timeout - elapsed_time)

                # Check if we've exceeded timeout
                if remaining_time <= 0:
                    process.terminate()
                    time.sleep(3)  # Give it a moment to terminate gracefully
                    if process.poll() is None:  # If still running
                        process.kill()  # Force kill
                    stdout_chunks.append(f"\nProcess reached time limit after {timeout} seconds.\n")
                    logger.info(f"\nProcess reached time limit after {timeout} seconds.\n")
                    break

                # Wait for output on either stream with timeout
                # select.select returns empty lists if the timeout elapses
                readable, _, _ = select.select(streams, [], [], min(1, remaining_time))

                # If nothing was read but process is still running, continue the loop
                if not readable and process.poll() is None:
                    continue

                # If nothing was read and process exited, exit loop
                if not readable and process.poll() is not None:
                    break

                for stream in readable:
                    line = stream.readline()
                    if not line:  # EOF
                        streams.remove(stream)
                        continue

                    # Handle stdout
                    if stream == process.stdout:
                        stdout_chunks.append(line)
                        logger.detail(line.rstrip())
                    # Handle stderr
                    else:
                        stderr_chunks.append(line)
                        logger.detail(line.rstrip())

            elapsed_time = time.time() - start_time
            progress_context.update(task, completed=elapsed_time)

        # Wait for process to complete (should already be done, but just in case)
        if process.poll() is None:
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                stderr_chunks.append("Process forcibly terminated after timeout\n")

        success = process.returncode == 0
        return success, "".join(stdout_chunks), "".join(stderr_chunks)

    except Exception as e:
        return False, "", f"Error executing {language} code: {str(e)}"


class ExecuterAgent(BaseAgent):
    """
    Execute the code and give analysis.

    Agent Input:

    Agent Output:
    """

    def __init__(self, config, manager, language, timeout, executer_llm_config, executer_prompt_template):
        super().__init__(config=config, manager=manager)
        assert language in ["bash", "python"]

        self.timeout = timeout
        self.language = language
        self.executer_llm_config = executer_llm_config

        if executer_prompt_template is not None:
            self.executer_prompt_template = executer_prompt_template
        elif self.executer_llm_config.template is not None:
            self.executer_prompt_template = self.executer_llm_config.template
        else:
            self.executer_prompt_template = None

        if self.executer_llm_config.multi_turn:
            self.executer_llm = init_llm(
                llm_config=self.executer_llm_config,
                agent_name=f"{language}_executer",
                multi_turn=self.executer_llm_config.multi_turn,
            )

        self.executer_prompt = ExecuterPrompt(
            llm_config=self.executer_llm_config, manager=manager, template=self.executer_prompt_template
        )

    def __call__(self, code_to_execute, code_to_analyze=None, task_description=None, data_prompt=None):

        self.manager.log_agent_start("ExecuterAgent: executing code and collecting stdout/stderr for evaluation.")

        if code_to_analyze is None:
            code_to_analyze = code_to_execute

        success, stdout, stderr = execute_code(code=code_to_execute, language=self.language, timeout=self.timeout)

        if not self.executer_llm_config.multi_turn:
            self.executer_llm = init_llm(
                llm_config=self.executer_llm_config,
                agent_name=f"{self.language}_executer",
                multi_turn=self.executer_llm_config.multi_turn,
            )

        # Build prompt for evaluating execution results
        prompt = self.executer_prompt.build(
            stdout=stdout,
            stderr=stderr,
            python_code=code_to_analyze,
            task_description=task_description,
            data_prompt=data_prompt,
        )

        # Query the LLM
        response = self.executer_llm.assistant_chat(prompt)

        # Parse the LLM response to extract decision and error summary
        decision, error_summary = self.executer_prompt.parse(response)

        # Log the decision and error summary
        logger.brief(f"Planner decision: {decision}")
        if error_summary:
            logger.info(f"Error summary: {error_summary}")

        self.manager.log_agent_end("ExecuterAgent: execution finished; planner decision logged.")

        return decision, error_summary, prompt, stderr, stdout
