import logging
import re
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
        formatted_info += f"Library Name: {tool_name}\n"
        formatted_info += f"Version: v{info['version']}\n"
        formatted_info += f"Description: {info['description']}\n"
        if info["features"]:
            formatted_info += "Key Features & Limitations:\n"
            for feature in info["features"]:
                formatted_info += f"- {feature}\n"
        formatted_info += "\n\n"
    return formatted_info


class ToolSelectorPrompt(BasePrompt):
    """Handles prompts for tool selection"""

    def default_template(self) -> str:
        """Default template for tool selection"""
        return """
You are a data science expert tasked with selecting the most appropriate ML library for a specific task.

### Task Description:
{task_description}

### Data Information:
{data_prompt}

### Available ML Libraries:
{tools_info}

IMPORTANT: Your response MUST follow this exact format:
---
SELECTED_LIBRARY: <write only the exact library name from the options above>
EXPLANATION: <provide your detailed reasoning>
---

Requirements for your response:
1. The SELECTED_LIBRARY must be exactly as shown in the available libraries list
2. Use the exact headers "SELECTED_LIBRARY:" and "EXPLANATION:"
3. Provide a clear, detailed explanation of why this library is the best choice
4. Consider the task requirements, data characteristics, and library features

Do not include any other formatting or additional sections in your response.
"""

    def build(self) -> str:
        """Build a prompt for the LLM to select appropriate library."""
        prompt = self.template.format(
            task_description=self.manager.task_description,
            data_prompt=self.manager.data_prompt,
            tools_info=_format_tools_info(registry.tools),
        )

        self.manager.save_and_log_states(
            content=prompt, save_name="tool_selector_prompt.txt", per_iteration=False, add_uuid=False
        )

        return prompt

    def parse(self, response: str) -> Tuple[str, str]:
        """
        Parse the library selection response from LLM with improved robustness.

        Args:
            response: The raw response from the LLM

        Returns:
            Tuple[str, str]: (selected_tool, explanation)
        """
        # Default values
        selected_tool = ""
        explanation = ""

        # Clean the response
        response = response.strip()

        # Try different parsing strategies
        # Strategy 1: Look for exact headers
        selected_library_match = re.search(r"SELECTED_LIBRARY:[\s]*(.+?)(?:\n|$)", response, re.IGNORECASE)
        explanation_match = re.search(
            r"EXPLANATION:[\s]*(.+?)(?=SELECTED_LIBRARY:|$)", response, re.IGNORECASE | re.DOTALL
        )

        # Strategy 2: Fallback to more lenient parsing
        if not selected_library_match:
            selected_library_match = re.search(
                r"(?:selected|chosen|recommended).*?(?:library|tool):[\s]*(.+?)(?:\n|$)", response, re.IGNORECASE
            )

        if not explanation_match:
            explanation_match = re.search(
                r"(?:explanation|reasoning|rationale):[\s]*(.+?)(?=$)", response, re.IGNORECASE | re.DOTALL
            )

        # Extract and clean the matches
        if selected_library_match:
            selected_tool = selected_library_match.group(1).strip()
        if explanation_match:
            explanation = explanation_match.group(1).strip()

        # Validate against available tools
        available_tools = set(registry.tools.keys())
        if selected_tool and selected_tool not in available_tools:
            # Try to find the closest match
            closest_match = min(available_tools, key=lambda x: len(set(x.lower()) ^ set(selected_tool.lower())))
            logger.warning(
                f"Selected tool '{selected_tool}' not in available tools. " f"Using closest match: '{closest_match}'"
            )
            selected_tool = closest_match

        # Final validation
        if not selected_tool:
            logger.error("Failed to extract selected tool from LLM response")
            selected_tool = list(registry.tools.keys())[0]  # Default to first available tool
            logger.warning(f"Defaulting to: {selected_tool}")

        if not explanation:
            logger.error("Failed to extract explanation from LLM response")
            explanation = "No explanation provided by the model."

        # Log the results
        self._log_results(response, selected_tool, explanation)

        return selected_tool

    def _log_results(self, response: str, selected_tool: str, explanation: str):
        """Log the parsing results."""
        self.manager.save_and_log_states(
            content=response, save_name="tool_selector_response.txt", per_iteration=False, add_uuid=False
        )
        self.manager.save_and_log_states(
            content=selected_tool, save_name="selected_tool.txt", per_iteration=False, add_uuid=False
        )
        self.manager.save_and_log_states(
            content=explanation, save_name="tool_selector_explanation.txt", per_iteration=False, add_uuid=False
        )
