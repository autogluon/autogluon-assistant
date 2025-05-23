import os
import uuid
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf

from .agents import CoderAgent, ExecuterAgent
from .llm import ChatLLMFactory
from .managers import Manager
from .utils import extract_archives


def save_iteration_state(
    iteration_folder,
    manager,
    stdout,
    stderr,
    planner_decision=None,
    planner_explanation=None,
):
    """
    Save the current state of the prompt generator and execution outputs to separate files.

    Args:
        iteration_folder (str): Path to the current iteration folder
        manager (Manager): Current prompt generator instance
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
        "user_input.txt": manager.user_input or "",
        "python_code.py": manager.python_code or "",
        "bash_script.sh": manager.bash_script or "",
        "error_message.txt": manager.error_message or "",
        "tutorial_prompt.txt": manager.tutorial_prompt or "",
        "data_prompt.txt": manager.data_prompt or "",
        "task_description.txt": manager.task_description or "",
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

    manager = Manager(
        input_data_folder=input_data_folder,
        output_folder=output_folder,
        config=config,
    )
    python_coder = CoderAgent(
        config=config, language="python", coding_mode="coder", llm_config=config.coder, prompt_template=None
    )  # TODO: support prompt_templates in arguments
    bash_coder = CoderAgent(
        config=config, language="bash", coding_mode="coder", llm_config=config.coder, prompt_template=None
    )  # TODO: support prompt_templates in arguments

    # Initialize executer agent
    # TODO: add executer_prompt_template in args
    executer = ExecuterAgent(
        config=config,
        language="bash",
        stream_output=stream_output,
        timeout=per_execution_timeout,
        executer_llm_config=config.executer,
        executer_prompt_template=None,
    )

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

        manager.step(user_input=user_input)

        # Generate code
        generated_python_code = python_coder(manager=manager)

        # TODO: ask manager the file path
        python_file_path = os.path.join(iteration_folder, "generated_code.py")

        manager.update_python_code(python_code=generated_python_code, python_file_path=python_file_path)

        # Generate bash code
        generated_bash_script = bash_coder(manager=manager)

        # TODO: ask manager the file path
        bash_file_path = os.path.join(iteration_folder, "execution_script.sh")

        manager.update_bash_script(bash_script=generated_bash_script, bash_file_path=bash_file_path)

        planner_decision, planner_error_summary, planner_prompt, stderr, stdout = executer(
            code_to_execute=generated_bash_script,
            code_to_analyze=generated_python_code,
            task_description=manager.task_description,
            data_prompt=manager.data_prompt,
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
            manager.update_error_message(error_message=error_message)

            # Let the user know we're continuing despite success
            print(f"Code generation failed in iteration {iteration}!")
        else:
            if planner_decision != "FINISH":
                print(f"###INVALID Planner Output: {planner_decision}###")
            print(f"Code generation successful after {iteration + 1} iterations")
            manager.update_error_message(error_message="")
            # Save the current state
            save_iteration_state(iteration_folder, manager, stdout, stderr)
            break

        # Save the current state
        save_iteration_state(
            iteration_folder,
            manager,
            stdout,
            stderr,
        )

        iteration += 1
        if iteration >= max_iterations:
            print(f"Warning: Reached maximum iterations ({max_iterations}) without success")

    token_usage_path = os.path.join(iteration_folder, "token_usage.json")
    print(f"Total token usage:\n{ChatLLMFactory.get_total_token_usage(save_path=token_usage_path)}")
