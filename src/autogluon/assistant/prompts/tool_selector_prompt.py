import logging
import re
from typing import Dict, List, Union

from ..constants import DEFAULT_LIBRARY
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
        formatted_info += "\n\n"
    return formatted_info


class ToolSelectorPrompt(BasePrompt):
    """Handles prompts for tool selection"""

    @classmethod
    def meta_instructions(cls) -> str:
        """
        Returns specific instructions for meta-prompting the Tool Selector template.
        """
        return """
The ToolSelectorPrompt selects the most appropriate machine learning library for a given task based on data characteristics and requirements.

Considerations for rewriting this template:
1. Focus on clear criteria for matching libraries to specific data types and task requirements
2. Include evaluation of library strengths and limitations for the particular use case
3. Consider computational efficiency requirements based on data size and available resources
4. Emphasize specific features of libraries that are most relevant to the task domain
5. Ensure the output format clearly identifies the selected tool with detailed justification
"""

    def default_template(self) -> str:
        """Default template for tool selection"""
        return """
You are a data science expert tasked with selecting and ranking the most appropriate ML libraries for a specific task.

### Task Description:
{task_description}

### Data Information:
{data_prompt}

### Available ML Libraries:
{tools_info}

IMPORTANT: Your response MUST follow this exact format:
---
EXPLANATION: <provide your detailed reasoning process for evaluating the libraries>

RANKED_LIBRARIES:
1. <first choice library name>
2. <second choice library name>
3. <third choice library name>
...
---

Requirements for your response:
1. First provide a detailed explanation of your reasoning process using the "EXPLANATION:" header
2. Then provide a ranking of libraries using the "RANKED_LIBRARIES:" header
3. The library names must be exactly as shown in the available libraries list
4. Provide a ranking of at least 3 libraries (if available)
5. In your explanation, analyze each library's strengths and weaknesses for this specific task
6. Consider the task requirements, data characteristics, and library features

Do not include any other formatting or additional sections in your response.
"""

    def _build(self, **kwargs) -> str:
        """Build a prompt for the LLM to select appropriate library.

        Args:
            **kwargs: Additional keyword arguments to customize the prompt building process
        """

        # Render the prompt using the variable provider with additional variables
        additional_vars = {"tools_info": _format_tools_info(registry.tools)}

        prompt = self.render(additional_vars)

        self.manager.save_and_log_states(
            content=prompt, save_name="tool_selector_prompt.txt", per_iteration=False, add_uuid=False
        )

        return prompt

    def parse(self, response: str) -> Union[List[str], str]:
        """
        Parse the library selection response from LLM with improved robustness.

        Args:
            response: The raw response from the LLM

        Returns:
            Union[List[str], str]: Either a prioritized list of tools or a single tool name
        """
        # Clean the response
        response = response.strip()

        # Extract explanation first
        explanation_match = re.search(
            r"EXPLANATION:[\s]*(.+?)(?=RANKED_LIBRARIES:|$)", response, re.IGNORECASE | re.DOTALL
        )

        if not explanation_match:
            explanation_match = re.search(
                r"(?:explanation|reasoning|rationale):[\s]*(.+?)(?=RANKED_LIBRARIES:|ranking|ranked|prioritized|priority|$)",
                response,
                re.IGNORECASE | re.DOTALL,
            )

        explanation = (
            explanation_match.group(1).strip() if explanation_match else "No explanation provided by the model."
        )

        # Strategy 1: Look for ranked libraries section
        ranked_libraries_section = re.search(r"RANKED_LIBRARIES:(.*?)$", response, re.IGNORECASE | re.DOTALL)

        # Strategy 2: Fallback to more lenient parsing
        if not ranked_libraries_section:
            ranked_libraries_section = re.search(
                r"(?:ranking|ranked|prioritized|priority).*?(?:libraries|tools):(.*?)$",
                response,
                re.IGNORECASE | re.DOTALL,
            )

        # Parse the ranked libraries
        prioritized_tools = []

        if ranked_libraries_section:
            # Get the list section
            ranked_section = ranked_libraries_section.group(1).strip()

            # Try to find numbered list items
            list_items = re.findall(r"^\s*\d+\.\s*(.+?)$", ranked_section, re.MULTILINE)

            if list_items:
                # Found a numbered list
                for item in list_items:
                    tool_name = item.strip()
                    if tool_name:
                        prioritized_tools.append(tool_name)
            else:
                # Try to find bullet points or just lines
                list_items = re.findall(r"(?:^|\n)\s*(?:[-*•])?\s*(.+?)(?:$|\n)", ranked_section)
                for item in list_items:
                    tool_name = item.strip()
                    if tool_name:
                        prioritized_tools.append(tool_name)

        # Validate against available tools and clean up
        available_tools = set(registry.tools.keys())
        validated_tools = []

        for tool in prioritized_tools:
            if tool in available_tools:
                validated_tools.append(tool)
            else:
                # Try to find the closest match
                closest_match = min(available_tools, key=lambda x: len(set(x.lower()) ^ set(tool.lower())))
                logger.warning(f"Tool '{tool}' not in available tools. Using closest match: '{closest_match}'")
                validated_tools.append(closest_match)

        # Final validation - if we couldn't parse any tools, default to original behavior
        if not validated_tools:
            logger.error("Failed to extract ranked tools from LLM response")
            default_tool = DEFAULT_LIBRARY
            logger.warning(f"Defaulting to single tool: {default_tool}")
            self._log_results(response, default_tool, explanation)
            return [default_tool]

        # Log the results
        tools_str = ", ".join(validated_tools)
        self._log_results(response, tools_str, explanation)

        return validated_tools

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
