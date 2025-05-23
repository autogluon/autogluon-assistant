import logging
from typing import List

from .base_prompt import BasePrompt

logger = logging.getLogger(__name__)


class DescriptionFileRetrieverPrompt(BasePrompt):
    """Handles prompts for description file identification"""

    def default_template(self) -> str:
        """Default template for description file identification"""
        return """
Given the data structure, please identify any files that appear to contain project descriptions, requirements, or task definitions.
Look for files like README, documentation files, or task description files.

### Data Structure
{data_prompt}

Format your response as follows:
Description Files: [list ONLY the absolute path, one per line]
"""

    def build(self, manager) -> str:
        """Build a prompt for the LLM to identify description files."""

        # Format the prompt using the template
        return self.template.format(
            data_prompt=manager.data_prompt,
        )

    def parse(self, response: str) -> List[str]:
        """Parse the LLM response to extract description files."""

        # Extract filenames from the response
        description_files = []
        lines = response.split("\n")
        in_files_section = False

        for line in lines:
            line_stripped = line.strip()

            if "description files:" in line_stripped.lower():
                in_files_section = True
                continue
            elif in_files_section and line_stripped:
                filename = line_stripped.strip("- []").strip()
                if filename:
                    description_files.append(filename)

        return description_files
