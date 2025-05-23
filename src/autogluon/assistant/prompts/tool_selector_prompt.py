import logging
from typing import Dict, Tuple

from ..tools_registry import registry
from .base_prompt import BasePrompt

logger = logging.getLogger(__name__)


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


class ToolSelectorPrompt(BasePrompt):
    """Handles prompts for tool selection"""

    def default_template(self) -> str:
        """Default template for tool selection"""
        return """
Given the following data science task, please select the most appropriate ML Library for this task. Consider:
1. The nature of the data (data modality, problem type, etc.)
2. The specific requirements of the task
3. The limitations or special features of each library

### Task Description
{task_description}

### Data Structure
{data_prompt}

### Available ML Library and Their Capabilities
{tools_info}

Format your response as follows:
Selected Library: [library name ONLY]
Explanation: [detailed explanation of why this library is the best choice, including specific features that match the task requirements]
"""

    def build(self, manager) -> str:
        """Build a prompt for the LLM to select appropriate library."""

        # Format the prompt using the template
        return self.template.format(
            task_description=manager.task_description,
            data_prompt=manager.data_prompt,
            tools_info=_format_tools_info(registry.tools),
        )

    def parse(self, response: str) -> Tuple[str, str]:
        """Parse the library selection response from LLM."""

        selected_tool = ""
        explanation = ""

        lines = response.split("\n")
        in_explanation = False

        for line in lines:
            line = line.strip()
            if line.lower().startswith("selected library:"):
                selected_tool = line.split(":", 1)[1].strip()
            elif line.lower().startswith("explanation:"):
                in_explanation = True
                explanation = line.split(":", 1)[1].strip()
            elif in_explanation and line:
                explanation += " " + line

        # Validate that we got both components
        # TODO: Fall back to default library?
        if not selected_tool:
            logger.warning("Failed to extract selected tool from LLM response")
            selected_tool = "Failed to extract selected tool from LLM response."

        if not explanation:
            logger.warning("Failed to extract explanation from LLM response")
            explanation = "Failed to extract explanation from LLM response."

        return selected_tool
