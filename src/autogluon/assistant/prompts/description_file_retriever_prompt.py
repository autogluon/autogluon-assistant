import logging
from typing import List

from .base_prompt import BasePrompt

logger = logging.getLogger(__name__)


class DescriptionFileRetrieverPrompt(BasePrompt):
    """Handles prompts for description file identification"""

    @classmethod
    def meta_instructions(cls) -> str:
        """
        Returns specific instructions for meta-prompting the Description File Retriever template.
        """
        return """
The DescriptionFileRetrieverPrompt identifies files that contain project descriptions, requirements, or task definitions.

Considerations for rewriting this template:
1. Focus on accurate identification of files with task-relevant descriptions
2. Include pattern recognition for common documentation file formats and naming conventions
3. Emphasize extraction of absolute file paths in the correct format
4. Prioritize README files, competition descriptions, and task specification documents
5. Ensure the output format is clean and correctly structured for downstream processing
"""

    def default_template(self) -> str:
        """Default template for description file identification"""
        return """
Given the data structure, please identify any files that appear to contain project descriptions, requirements, or task definitions.
Look for files like README, documentation files, or task description files.

### Data Structure
{data_prompt}

Format your response as follows, do not give explanations:
Description Files: [list ONLY the absolute path, one per line]
"""

    def _build(self, **kwargs) -> str:
        """Build a prompt for the LLM to identify description files.

        Args:
            **kwargs: Additional keyword arguments to customize the prompt building process
        """

        # Render the prompt using the variable provider
        prompt = self.render()

        self.manager.save_and_log_states(
            content=prompt, save_name="description_file_retriever_prompt.txt", per_iteration=False, add_uuid=False
        )

        return prompt

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

        self.manager.save_and_log_states(
            content=response, save_name="description_file_retriever_response.txt", per_iteration=False, add_uuid=False
        )
        self.manager.save_and_log_states(
            content=description_files, save_name="description_files.txt", per_iteration=False, add_uuid=False
        )

        return description_files
