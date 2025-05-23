import logging
from typing import Optional

from .base_prompt import BasePrompt

logger = logging.getLogger(__name__)


class TaskDescriptorPrompt(BasePrompt):
    """Handles prompts for task description generation"""

    def default_template(self) -> str:
        """Default template for task description generation"""
        return """
Based ONLY on the information explicitly stated in the provided data structure and description files, provide a condensed description of the data science task. Include only details that are directly mentioned in the source materials. Do not add assumptions or infer unstated information.

### Data Structure:
(IMPORTANT: The metadata of example files in Data Structure may not be representative - do not make assumptions about data statistics based on examples.)
{data_prompt}

### Description File Contents:
{description_file_contents}
"""

    def build(self, manager) -> str:
        """Build a prompt for the LLM to generate task description."""

        file_contents = []
        for filename in manager.description_files:
            try:
                with open(filename, "r") as f:
                    content = f.read()
                file_contents.append(f"File: {filename}\nContent: {content}\n")
            except Exception as e:
                logger.warning(f"Could not read content of {filename}: {e}")
                continue

        description_file_contents = (
            "\n".join(file_contents) if file_contents else "No description file contents could be read."
        )

        # Format the prompt using the template
        return self.template.format(
            data_prompt=manager.data_prompt,
            description_file_contents=description_file_contents,
        )

    def parse(self, response: str) -> Optional[str]:
        """
        Parse the LLM response to extract task description.
        
        Args:
            response: Raw LLM response
            
        Returns:
            str: Parsed task description or error message
        """
        # For task description, we typically want the entire response
        # as it should be the complete task description
        if response and response.strip():
            return response.strip()
        else:
            return "Failed to generate task description from LLM response."
