import logging

from ..prompts import ExecuterPrompt
from .base_agent import BaseAgent
from .utils import init_llm

logger = logging.getLogger(__name__)


def execute_code(code, language, stream_output, timeout):
    """
    Execute code with real-time output streaming and timeout.
    Args:
        code (str): The code to execute (Python code or bash script)
        language (str): The language to execute ("python" or "bash")
        stream_output (bool): Whether to stream output in real-time
        timeout (int): Maximum execution time in seconds before terminating the process
    Returns:
        tuple: (success, stdout, stderr)
    """
    import select
    import subprocess
    import sys
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

        stdout_chunks = []
        stderr_chunks = []

        # Set up tracking of both output streams
        streams = [process.stdout, process.stderr]

        # Track start time for timeout
        start_time = time.time()

        while streams:
            # Calculate remaining time
            elapsed_time = time.time() - start_time
            remaining_time = max(0, timeout - elapsed_time)

            # Check if we've exceeded timeout
            if remaining_time == 0:
                process.terminate()
                time.sleep(10)  # Give it a moment to terminate gracefully
                if process.poll() is None:  # If still running
                    process.kill()  # Force kill
                stdout_chunks.append(f"\nProcess reached time limit after {timeout} seconds\n")
                if stream_output:
                    sys.stdout.write(f"\nProcess reached time limit after {timeout} seconds\n")
                    sys.stdout.flush()
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
                    if stream_output:
                        sys.stdout.write(line)
                        sys.stdout.flush()
                # Handle stderr
                else:
                    stderr_chunks.append(line)
                    if stream_output:
                        sys.stderr.write(line)
                        sys.stderr.flush()

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

    def __init__(self, config, manager, language, stream_output, timeout, executer_llm_config, executer_prompt_template):
        super().__init__(config=config, manager=manager)
        assert language in ["bash", "python"]

        self.stream_output = stream_output
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
        if code_to_analyze is None:
            code_to_analyze = code_to_execute

        success, stdout, stderr = execute_code(
            code=code_to_execute, language=self.language, stream_output=self.stream_output, timeout=self.timeout
        )

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
        logger.info(f"Planner decision: {decision}")
        if error_summary:
            logger.info(f"Error summary: {error_summary}")

        return decision, error_summary, prompt, stderr, stdout
