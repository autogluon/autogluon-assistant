import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from ..llm import ChatLLMFactory
from ..tools_registry import registry

logger = logging.getLogger(__name__)


def find_description_files(data_prompt: str, llm) -> Tuple[List[str], str]:
    """
    Step 1: Use LLM to identify potential description files from the data prompt.
    Only identifies files, does not read content.

    Args:
        data_prompt: Text string containing data prompt
        llm: Initialized LLM model

    Returns:
        Tuple[List[str], str]: (List of identified description filenames, Analysis explanation)
    """
    find_descriptions_prompt = f"""
Given this data prompt:

{data_prompt}

Please identify any files that appear to contain project descriptions, requirements, or task definitions.
Look for files like README, documentation files, or task description files.

Format your response as follows:
Description Files: [list ONLY the absolute path, one per line]
Explanation: [explain why these files were identified as description files]
    """

    analysis = llm.assistant_chat(find_descriptions_prompt)

    # Extract filenames from the response
    description_files = []
    lines = analysis.split("\n")
    in_files_section = False

    for line in lines:
        line = line.strip()
        if "description files:" in line.lower():
            in_files_section = True
            continue
        elif "explanation:" in line.lower():
            break
        elif in_files_section and line:
            filename = line.strip("- []").strip()
            if filename:
                description_files.append(filename)

    return description_files, analysis


def select_tool(data_prompt: str, description: str, llm) -> Tuple[str, str]:
    """
    Use LLM to select the most appropriate tool based on the data description and available tools.

    Args:
        data_prompt: Text string containing data prompt
        description: Description of the task/data from previous analysis
        llm: Initialized LLM model

    Returns:
        Tuple[str, str]: (Selected tool name, Explanation for the selection)
    """
    # Get all available tools and their information
    tools_info = registry.tools

    # Construct prompt for tool selection
    tool_selection_prompt = f"""
Given the following data science task:

Data Description:
{data_prompt}

Task Analysis:
{description}

Available tools and their capabilities:

{_format_tools_info(tools_info)}

Please select the most appropriate tool for this task. Consider:
1. The nature of the data (tabular, time series, multimodal, etc.)
2. The specific requirements of the task
3. Any limitations or special features of each tool

Format your response as follows:
Selected Tool: [tool name ONLY]
Explanation: [detailed explanation of why this tool is the best choice, including specific features that match the task requirements]
"""

    # Get LLM's tool selection and reasoning
    response = llm.assistant_chat(tool_selection_prompt)

    # Parse the response
    selected_tool = ""
    explanation = ""

    lines = response.split("\n")
    in_explanation = False

    for line in lines:
        line = line.strip()
        if line.lower().startswith("selected tool:"):
            selected_tool = line.split(":", 1)[1].strip()
        elif line.lower().startswith("explanation:"):
            in_explanation = True
            explanation = line.split(":", 1)[1].strip()
        elif in_explanation:
            explanation += " " + line

    # Validate selected tool exists in registry
    if not registry.get_tool(selected_tool):
        logger.warning(f"Selected tool '{selected_tool}' not found in registry")
        raise ValueError(
            f"Selected tool '{selected_tool}' is not available in the tools registry"
        )

    return selected_tool, explanation


def _format_tools_info(tools_info: Dict) -> str:
    """
    Format tools information for the prompt.

    Args:
        tools_info: Dictionary containing tool information

    Returns:
        str: Formatted string of tool information
    """
    formatted_info = ""
    for tool_name, info in tools_info.items():
        formatted_info += f"Tool Name: \n{tool_name}\n"
        formatted_info += f"Version: v{info['version']}\n"
        formatted_info += f"Description: {info['description']}\n"
        if info["features"]:
            formatted_info += "Special features/limitations:\n"
            for feature in info["features"]:
                formatted_info += f"- {feature}\n"
        formatted_info += "\n"
    return formatted_info


def generate_task_description(
    data_prompt: str, description_files: List[str], description_analysis: str, llm
) -> str:
    """
    Step 2: Read content of identified files and generate task description.

    Args:
        data_prompt: Text string containing data prompt
        description_files: List of description filenames from step 1
        description_analysis: Analysis from step 1
        llm: Initialized LLM model

    Returns:
        str: Generated task description
    """
    try:
        # Read content of identified description files
        file_contents = []
        for filename in description_files:
            try:
                with open(filename, "r") as f:
                    content = f.read()
                file_contents.append(f"File: {filename}\nContent: {content}\n")
            except Exception as e:
                logger.warning(f"Could not read content of {filename}: {e}")
                continue

        description_context = (
            "\n".join(file_contents)
            if file_contents
            else "No description file contents could be read."
        )

        task_prompt = f"""
Based on this data prompt and description files:

Data Prompt:
(IMPORTANT: The metadata of example files in Data Prompt may not be representative - do not make assumptions about data statistics based on examples.)
{data_prompt}

Description File Analysis:
{description_analysis}

Description File Contents:
{description_context}

Based ONLY on the information explicitly stated in the provided data prompt, description files, and analysis, provide a condensed description of the data science task. Include only details that are directly mentioned in the source materials.
Do not add assumptions or infer unstated information.
        """

        response = llm.assistant_chat(task_prompt)
        return response

    except Exception as e:
        logger.error(f"Error in generating task description: {e}")
        return f"Error generating task description: {str(e)}"


def wrap_task_description(
    task_description: str, output_folder: str, tool_name: str, registry
) -> str:
    """
    Wraps the task description with standard instructions and tool-specific requirements.

    Args:
        task_description: Generated description of the data science task
        output_folder: Path where outputs should be saved
        tool_name: Name of the selected tool
        registry: Tool registry containing tool-specific information

    Returns:
        str: Complete task prompt including general and tool-specific instructions
    """
    # Get tool-specific template and requirements if they exist
    tool_info = registry.get_tool(tool_name)
    if not tool_info:
        raise ValueError(f"Tool {tool_name} not found in registry")

    # Get tool-specific template or use default format
    tool_prompt = tool_info.get("prompt_template", "")
    if isinstance(tool_prompt, list):
        tool_prompt = "\n".join(tool_prompt)

    return f"""
As an AutoML Agent, you will be given a folder containing data and description files. Please generate Python code using {tool_name} to train a predictor and make predictions on test data. Follow these specifications:

ONLY save files to the working directory: {output_folder}.

1. Data preprocessing:
   - Remove training data samples without valid labels (drop NA values from training dataset ONLY, NOT from test dataset) unless explicitly instructed otherwise.
   - Remove the unneccesary index column (if applicable)

2. Model training:
   - Use {tool_name} with appropriate parameters for the task
   - If a model is trained, save it in a folder with random timestamp within {output_folder}

3. Prediction:
   - Make predictions on the test data
   - Save the predicted results to {output_folder}, result file name should be "results", the format and extension should be same as the test data file
   - Output column names must exactly match those in the training or sample submission files without adding "predicted_" prefixes or creating any new columns.

4. Documentation:
   - Add a brief docstring at the beginning of the script explaining its purpose and usage
   - Also include additional installation steps with comments at the beginning of the script
   - Include comments explaining any complex operations or design decisions

5. Others:
   - To avoid DDP errors, wrap the code in: if __name__ == "__main__":
   - Ensure errors are propagated up and not silently caught - do not use try/except blocks unless you explicitly re-raise the exception.

{tool_prompt}

Please provide the complete Python script that accomplishes these tasks, ensuring it's ready to run given the appropriate data inputs.

Task Description: {task_description}
"""


def generate_task_prompt(data_prompt: str, output_folder: str, llm_config) -> str:
    """
    Main function to generate task prompt following two-step process.

    Args:
        data_prompt: Text string containing data prompt
        output_folder: Path to the output folder
        llm_config: Configuration for the LLM model

    Returns:
        str: Generated task prompt
    """
    # Ensure output folder exists
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # TODO: use one conversation for both tasks?
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    llm_find_description_files = ChatLLMFactory.get_chat_model(
        llm_config, session_name=f"description_finder_{timestamp}"
    )
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    llm_generate_task_description = ChatLLMFactory.get_chat_model(
        llm_config, session_name=f"task_generator_{timestamp}"
    )
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    llm_tool_selection = ChatLLMFactory.get_chat_model(
        llm_config, session_name=f"tool_selector_{timestamp}"
    )

    # Step 1: Find description files (just identifies files, doesn't read content)
    description_files, description_analysis = find_description_files(
        data_prompt, llm_find_description_files
    )
    logger.info(
        f"Found {len(description_files)} potential description files: {description_files}"
    )

    # Step 2: Generate task description (includes reading file contents)
    task_description = generate_task_description(
        data_prompt,
        description_files,
        description_analysis,
        llm_generate_task_description,
    )

    # Step 3: Select the ML tool to use
    selected_tool, explanation = select_tool(
        data_prompt=data_prompt, description=task_description, llm=llm_tool_selection
    )

    task_description = wrap_task_description(
        task_description=task_description,
        output_folder=output_folder,
        tool_name=selected_tool,
        registry=registry,
    )

    # Save results in separate files
    # Save description file names
    files_path = os.path.join(output_folder, "description_files.txt")
    with open(files_path, "w") as f:
        for filename in description_files:
            f.write(f"{filename}\n")
    logger.info(f"Description files list saved to: {files_path}")

    # Save description analysis
    analysis_path = os.path.join(output_folder, "description_analysis.txt")
    with open(analysis_path, "w") as f:
        f.write(description_analysis)
    logger.info(f"Description analysis saved to: {analysis_path}")

    # Save generated task description
    task_path = os.path.join(output_folder, "task_description.txt")
    with open(task_path, "w") as f:
        f.write(task_description)
    logger.info(f"Generated task description saved to: {task_path}")

    # Save tool selection
    tool_path = os.path.join(output_folder, "tool_selection.txt")
    with open(tool_path, "w") as f:
        f.write(selected_tool)
        f.write("\n\n")
        f.write(explanation)
    logger.info(f"Tool selection log is saved to: {tool_path}")

    return task_description, selected_tool
