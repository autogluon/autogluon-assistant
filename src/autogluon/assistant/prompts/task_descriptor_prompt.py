import logging
from typing import Optional

from .base_prompt import BasePrompt

logger = logging.getLogger(__name__)


class TaskDescriptorPrompt(BasePrompt):
    """Handles prompts for task description generation"""
    
    @classmethod
    def meta_template(cls) -> str:
        """
        Returns specific instructions for meta-prompting the Task Descriptor template.
        """
        return """
The TaskDescriptorPrompt analyzes data structure and description files to generate a precise technical description of the machine learning task.

Considerations for rewriting this template:
1. Focus on accurate identification of problem type and evaluation metrics
2. Include guidance for extracting key data characteristics relevant to model selection
3. Emphasize clarity in describing input/output formats and requirements
4. Prioritize extracting technical specifications over general descriptions
5. Ensure the resulting description captures all critical task constraints and objectives
"""
    
    def default_template(self) -> str:
        """Default template for task description generation"""
        return """
Based ONLY on the information explicitly stated in the provided data structure and description files, provide a condensed and precise description of the data science task. Include only details that are directly mentioned in the source materials. Do not add assumptions or infer unstated information.

Be very clear about the problem type (e.g. audio classification/image regression/seq-to-seq generation/etc.), input format, and prediction output format.

### Data Structure:
(IMPORTANT: The metadata of example files in Data Structure may not be representative - do not make assumptions about data statistics based on examples.)
{data_prompt}

### Description File Contents:
{description_file_contents_truncate_end_16384}
"""

    def get_description_files_contents(self, to_show=False):
        # if to show is false, it is used by LLM for summarization
        # if to show is true, it is shown in prompts for coder, etc., then we keep only the file contents

        file_contents = []
        for filename in self.manager.description_files:
            try:
                with open(filename, "r") as f:
                    content = f.read()
                if to_show:
                    file_contents.append(content)
                else:
                    file_contents.append(f"File: {filename}\nContent: {content}")
            except Exception as e:
                logger.warning(f"Could not read content of {filename}: {e}")
                continue

        description_file_contents = (
            "\n\n".join(file_contents) if file_contents else "No description file contents could be read."
        )
        return description_file_contents

    def _build(self, **kwargs) -> str:
        """Build a prompt for the LLM to generate task description.
        
        Args:
            **kwargs: Additional keyword arguments to customize the prompt building process
        """
        

        description_file_contents = self.get_description_files_contents()

        # Render the prompt using the variable provider with additional variables
        additional_vars = {"description_file_contents": description_file_contents}

        prompt = self.render(additional_vars)

        self.manager.save_and_log_states(
            content=prompt, save_name="task_descriptor_prompt.txt", per_iteration=False, add_uuid=False
        )

        return prompt

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
            task_description = response.strip()
        else:
            task_description = "Failed to generate task description from LLM response."

        self.manager.save_and_log_states(
            content=response, save_name="task_descriptor_response.txt", per_iteration=False, add_uuid=False
        )
        self.manager.save_and_log_states(
            content=task_description, save_name="task_description.txt", per_iteration=False, add_uuid=False
        )

        return task_description
