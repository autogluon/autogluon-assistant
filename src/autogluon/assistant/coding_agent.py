import os
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf

from .coder import generate_coder, write_code_script, write_retrieved_context
from .llm import ChatLLMFactory
from .planner import get_planner
from .prompt import PromptGenerator, write_prompt_to_file
from .utils import extract_archives


def execute_bash_script(bash_script, stream_output=True, timeout=3600 * 6):
    """
    Execute bash script with real-time output streaming and timeout.

    Args:
        bash_script (str): The bash script to execute
        stream_output (bool): Whether to stream output in real-time
        timeout (int): Maximum execution time in seconds before terminating the process

    Returns:
        tuple: (success, stdout, stderr)
    """
    import select
    import time

    try:
        process = subprocess.Popen(
            ["bash", "-c", bash_script],
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
                stderr_chunks.append(f"\nProcess timed out after {timeout} seconds\n")
                if stream_output:
                    sys.stderr.write(f"\nProcess timed out after {timeout} seconds\n")
                    sys.stderr.flush()
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
        return False, "", f"Error executing bash script: {str(e)}"


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
    output_folder=None,
    tutorial_link=None,
    config_path=None,
    max_iterations=5,
    need_user_input=False,
    initial_user_input=None,
    extract_archives_to=None,
):
    # Get the directory of the current file
    current_file_dir = Path(__file__).parent

    if output_folder is None or not output_folder:
        working_dir = os.path.join(current_file_dir.parent.parent.parent, "runs")
        # Get current date in YYYYMMDD format
        current_date = datetime.now().strftime("%Y%m%d")
        # Generate a random UUID4
        random_uuid = uuid.uuid4()
        # Create the folder name using the pattern
        folder_name = f"mlzero-{current_date}-{random_uuid}"

        # Create the full path for the new folder
        output_folder = os.path.join(working_dir, folder_name)

    # Create output directory
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    if extract_archives_to is not None:
        if extract_archives_to and extract_archives_to != input_data_folder:
            import shutil

            # Create the destination directory if it doesn't exist
            os.makedirs(extract_archives_to, exist_ok=True)

            # Walk through all files and directories in the source folder
            for root, dirs, files in os.walk(input_data_folder):
                # Calculate the relative path from the source folder
                rel_path = os.path.relpath(root, input_data_folder)

                # Create the corresponding directory structure in the destination
                if rel_path != ".":
                    dest_dir = os.path.join(extract_archives_to, rel_path)
                    os.makedirs(dest_dir, exist_ok=True)
                else:
                    dest_dir = extract_archives_to

                # Copy all files in the current directory
                for file in files:
                    src_file = os.path.join(root, file)
                    dest_file = os.path.join(dest_dir, file)
                    shutil.copy2(src_file, dest_file)  # copy2 preserves metadata

            input_data_folder = extract_archives_to
            print(
                f"Note: we strongly recommend using data without archived files. Extracting archived files under {input_data_folder}..."
            )
            extract_archives(input_data_folder)

    # Always load default config first
    default_config_path = current_file_dir / "configs" / "default.yaml"
    if not default_config_path.exists():
        raise FileNotFoundError(f"Default config file not found: {default_config_path}")

    config = OmegaConf.load(default_config_path)

    # If config_path is provided, merge it with the default config
    if config_path is not None:
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        user_config = OmegaConf.load(config_path)
        config = OmegaConf.merge(config, user_config)

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
                print(f"\nPrevious iteration files are in: {os.path.join(output_folder, f'iteration_{iteration-1}')}")
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
            print(f"Code generation failed in iteration {iteration}!")
        else:
            if planner_decision != "FINISH":
                print(f"###INVALID Planner Output: {planner_decision}###")
            print(f"Code generation successful after {iteration + 1} iterations")
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
            print(f"Warning: Reached maximum iterations ({max_iterations}) without success")

    token_usage_path = os.path.join(iteration_folder, "token_usage.json")
    print(f"Total token usage:\n{ChatLLMFactory.get_total_token_usage(save_path=token_usage_path)}")
