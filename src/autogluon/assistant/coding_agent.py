import logging
import os
import select
import subprocess
import sys
import time
from pathlib import Path

from omegaconf import OmegaConf
from rich import print
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
)

from .coder import generate_coder, write_code_script, write_retrieved_context
from .llm import ChatLLMFactory
from .planner import get_planner
from .prompt import PromptGenerator, write_prompt_to_file

logger = logging.getLogger(__name__)


def execute_bash_script(bash_script: str, stream_output: bool = True, timeout: float = 3600 * 6):
    """
    Execute bash script with real-time output streaming and timeout and show a linear timeout progress bar.

    Args:
        bash_script (str): The bash script to execute.
        stream_output (bool): Whether to stream stdout/stderr via logger.model_info.e
        timeout (float): Maximum execution time in seconds before terminating the process.

    Returns:
        tuple: (success: bool, stdout: str, stderr: str)
    """
    try:
        process = subprocess.Popen(
            ["bash", "-c", bash_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        stdout_chunks, stderr_chunks = [], []
        streams = [process.stdout, process.stderr]
        start_time = time.time()

        with Progress(
            TextColumn(f"[cyan]Execution of maximum remaining time ({int(timeout)}s)[/]"),
            BarColumn(bar_width=None),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            transient=True,
        ) as progress:
            task = progress.add_task("", total=timeout)

            while streams:
                elapsed = time.time() - start_time
                progress.update(task, completed=min(elapsed, timeout))

                if elapsed >= timeout:
                    process.terminate()
                    time.sleep(1)
                    if process.poll() is None:
                        process.kill()
                    stderr_chunks.append(f"\nProcess timed out after {timeout} seconds\n")
                    if stream_output:
                        sys.stderr.write(f"\nProcess timed out after {timeout} seconds\n")
                        sys.stderr.flush()
                    break

                readable, _, _ = select.select(streams, [], [], 1.0)
                if not readable and process.poll() is None:
                    continue
                if not readable and process.poll() is not None:
                    break

                for s in readable:
                    line = s.readline()
                    if not line:
                        streams.remove(s)
                        continue
                    if s is process.stdout:
                        stdout_chunks.append(line)
                        if stream_output:
                            logger.model_info(line.rstrip())
                    else:
                        stderr_chunks.append(line)
                        if stream_output:
                            logger.model_info(line.rstrip())

            progress.update(task, completed=timeout)

        if process.poll() is None:
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                stderr_chunks.append("Process forcibly terminated after timeout\n")

        success = process.returncode == 0
        return success, "".join(stdout_chunks), "".join(stderr_chunks)

    except Exception as e:
        return False, "", f"Error executing bash script: {e}"


def save_iteration_state(
    iteration_folder,
    prompt_generator,
    stdout,
    stderr,
    planner_decision=None,
    planner_explanation=None,
):
    """
    Save the current state of the prompt generator and execution outputs to separate files.

    Args:
        iteration_folder (str): Path to the current iteration folder
        prompt_generator (PromptGenerator): Current prompt generator instance
        stdout (str): Standard output from execution
        stderr (str): Standard error from execution
        planner_decision (str, optional): Decision from log evaluation (planner agent)
        planner_explanation (str, optional): Explanation from log evaluation (planner agent)
    """
    # Create a states subfolder
    states_folder = os.path.join(iteration_folder, "states")
    os.makedirs(states_folder, exist_ok=True)

    # Save each state component to a separate file
    state_files = {
        "user_input.txt": prompt_generator.user_input or "",
        "python_code.py": prompt_generator.python_code or "",
        "bash_script.sh": prompt_generator.bash_script or "",
        "error_message.txt": prompt_generator.error_message or "",
        "tutorial_prompt.txt": prompt_generator.tutorial_prompt or "",
        "data_prompt.txt": prompt_generator.data_prompt or "",
        "task_prompt.txt": prompt_generator.task_prompt or "",
        "stdout.txt": stdout or "",
        "stderr.txt": stderr or "",
    }

    for filename, content in state_files.items():
        file_path = os.path.join(states_folder, filename)
        with open(file_path, "w") as f:
            f.write(content)


def run_agent(
    input_data_folder,
    tutorial_link,
    output_folder,
    config_path,
    max_iterations=5,
    need_user_input=False,
    initial_user_input=None,
):
    # Load config from YAML and merge with default
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config = OmegaConf.load(config_path)

    stream_output = config.stream_output
    per_execution_timeout = config.per_execution_timeout

    prompt_generator = PromptGenerator(
        input_data_folder=input_data_folder,
        output_folder=output_folder,
        config=config,
    )
    python_coder = generate_coder(llm_config=config.coder, tutorial_link_for_rag=tutorial_link)
    bash_coder = generate_coder(llm_config=config.coder, tutorial_link_for_rag=tutorial_link)

    # Initialize log evaluation agent
    planner = get_planner(config.planner)

    iteration = 0
    while iteration < max_iterations:
        print(f"Starting iteration {iteration}!")

        # Create iteration subfolder
        iteration_folder = os.path.join(output_folder, f"iteration_{iteration}")
        os.makedirs(iteration_folder, exist_ok=True)

        user_input = None
        # Use initial user input at first iter
        if iteration == 0:
            user_input = initial_user_input
        # Get per iter user inputs if needed
        if need_user_input:
            if iteration > 0:
                previous_path = os.path.join(output_folder, f"iteration_{iteration-1}")
                print(f"\n[bold green]Previous iteration files are in:[/bold green] {previous_path}")
            if not user_input:
                user_input = ""
            user_input += input("Enter your inputs for this iteration (press Enter to skip): ")

        prompt_generator.step(user_input=user_input)

        # Generate and save the coding prompt
        coding_prompt = prompt_generator.get_coding_prompt()
        coding_prompt_path = os.path.join(iteration_folder, "coding_prompt.txt")
        write_prompt_to_file(coding_prompt, coding_prompt_path)

        # Generate code
        generated_content = python_coder(prompt=coding_prompt, language="python")
        generated_python_code = generated_content["code_script"]

        # Save the python code
        python_file_path = os.path.join(iteration_folder, "generated_code.py")
        write_code_script(generated_python_code, python_file_path)

        # Write retrieved context if present
        if "retrieved_context" in generated_content:
            output_context_path = os.path.join(iteration_folder, "retrieved_context.txt")
            write_retrieved_context(generated_content["retrieved_context"], output_context_path)

        prompt_generator.update_python_code(python_code=generated_python_code)

        # Generate and save the execution prompt
        execution_prompt = prompt_generator.get_execution_prompt(python_file_path=python_file_path)
        execution_prompt_path = os.path.join(iteration_folder, "execution_prompt.txt")
        write_prompt_to_file(execution_prompt, execution_prompt_path)

        # Generate bash code
        generated_bash_script = bash_coder(prompt=execution_prompt, language="bash")["code_script"]

        # Save the bash code
        bash_file_path = os.path.join(iteration_folder, "execution_script.sh")
        write_code_script(generated_bash_script, bash_file_path)

        prompt_generator.update_bash_script(bash_script=generated_bash_script)

        # Attempt to execute the generated code
        success, stdout, stderr = execute_bash_script(
            bash_script=generated_bash_script, stream_output=stream_output, timeout=per_execution_timeout
        )

        # Initialize log evaluation variables
        planner_decision = None
        planner_error_summary = None

        # Even though execution succeeded, evaluate logs to check for issues or poor performance
        planner_decision, planner_error_summary, planner_prompt = planner(
            stdout=stdout,
            stderr=stderr,
            python_code=generated_python_code,
            task_prompt=prompt_generator.task_prompt,
            data_prompt=prompt_generator.data_prompt,
        )

        # Save planner results
        planner_decision_path = os.path.join(iteration_folder, "planner_decision.txt")
        with open(planner_decision_path, "w") as f:
            f.write(f"planner_decision: {planner_decision}\n\nplanner_error_summary: {planner_error_summary}")
        planner_prompt_path = os.path.join(iteration_folder, "planner_prompt.txt")
        with open(planner_prompt_path, "w") as f:
            f.write(f"planner_prompt: {planner_prompt}")

        if planner_decision == "FIX":
            # Add suggestions to the error message to guide next iteration
            error_message = f"stderr: {stderr}\n\n" if stderr else ""
            error_message += (
                f"Error summary from planner (the error can appear in stdout if it's catched): {planner_error_summary}"
            )
            prompt_generator.update_error_message(error_message=error_message)

            # Let the user know we're continuing despite success
            print(f"[bold red]Code generation failed in iteration[/bold red] {iteration}!")
        else:
            if planner_decision != "FINISH":
                print(f"[bold red]###INVALID Planner Output:[/bold red] {planner_decision}###")
            print(f"[bold green]Code generation successful after[/bold green] {iteration + 1} iterations")
            prompt_generator.update_error_message(error_message="")
            # Save the current state
            save_iteration_state(iteration_folder, prompt_generator, stdout, stderr)
            break

        # Save the current state
        save_iteration_state(
            iteration_folder,
            prompt_generator,
            stdout,
            stderr,
        )

        iteration += 1
        if iteration >= max_iterations:
            print(
                f"[bold yellow]Warning: Reached maximum iterations ([/bold yellow]{max_iterations}[bold yellow]) without success[/bold yellow]"
            )

    token_usage_path = os.path.join(iteration_folder, "token_usage.json")
    usage = ChatLLMFactory.get_total_token_usage(save_path=token_usage_path)
    total = usage["total"]
    logger.brief(
        f"Total tokens — input: {total['total_input_tokens']}, "
        f"output: {total['total_output_tokens']}, "
        f"sum: {total['total_tokens']}"
    )

    logger.info(f"Full token usage detail:\n{usage}")
