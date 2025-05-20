import os
from textwrap import dedent


def generate_execution_prompt(
    output_folder,
    python_file_path,
    create_venv=True,
    install_packages=False,
    previous_bash=None,
    previous_python=None,
    error_message=None,
    current_python=None,
    max_error_message_length: int = 2048,
):
    """
    Generate a prompt for an LLM to create a simplified bash script for environment setup and code execution.

    Args:
        output_folder (str): Path to the project folder
        python_file_path (str): Absolute path to the Python file that needs to be executed
        create_venv (bool): Whether to create a new virtual environment or use current environment
        previous_bash (str, optional): Previous bash script that caused errors
        previous_python (str, optional): Previous Python code in the bash script
        error_message (str, optional): Previous error message to help with debugging
        current_python (str, optional): Current Python code to be executed

    Returns:
        str: Formatted prompt for the LLM
    """
    os.makedirs(output_folder, exist_ok=True)

    # Truncate error message if needed
    if len(error_message) > max_error_message_length:
        error_message = (
            error_message[: max_error_message_length // 2]
            + "\n...(truncated)\n"
            + error_message[-max_error_message_length // 2 :]
        )

    # Build the core instructions
    instructions = []
    if create_venv:
        instructions.extend(
            [
                f"Create and configure a conda environment in {output_folder}:",
                "- Python version: 3.11",
                "- Activate the environment",
                "- Install required packages",
            ]
        )
    elif install_packages:
        instructions.append(
            "The environment may not be fully configured. Install any packages required in the python code."
        )
    else:
        instructions.append("The environment is already configured. Do not install or update any package.")

    instructions.append(f"Execute the Python script: {python_file_path}")

    # Build the prompt with optional context
    prompt_parts = [
        "Generate a minimal bash script that will:",
        "\n".join(f"{i+1}. {instr}" for i, instr in enumerate(instructions)),
    ]

    if current_python:
        prompt_parts.append(
            dedent(
                f"""
            Current Python code:
            ```python
            {current_python}
            ```
        """
            ).strip()
        )

    if error_message:
        prompt_parts.append(f"Previous error:\n{error_message}")

    if previous_bash and error_message:
        prompt_parts.append(
            dedent(
                f"""
            Previous failed bash script:
            ```bash
            {previous_bash}
            ```
        """
            ).strip()
        )

    if previous_python and error_message:
        prompt_parts.append(
            dedent(
                f"""
            Previous Python code:
            ```python
            {previous_python}
            ```
        """
            ).strip()
        )

    # Add final instructions
    prompt_parts.append(
        dedent(
            """
        Notes:
        - Generate a minimal, executable bash script
        - Focus on essential commands only
        - Handle common environment and package only if there were errors
    """
        ).strip()
    )

    return "\n\n".join(prompt_parts)
