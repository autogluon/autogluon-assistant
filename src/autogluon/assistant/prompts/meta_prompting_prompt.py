"""
Meta-prompting prompt class to dynamically rewrite other prompts based on current task.

This module provides the MetaPromptingPrompt class which can analyze the current
task and variables to generate customized prompts for other agents.
"""

import logging
from typing import Dict

from .base_prompt import BasePrompt

logger = logging.getLogger(__name__)


class MetaPromptingPrompt(BasePrompt):
    """Handles meta-prompting for customizing other agent prompts"""

    @classmethod
    def meta_instructions(cls) -> str:
        """
        Note: Meta-prompting is not applied to the meta-prompting template itself.
        """
        return NotImplementedError

    def default_template(self) -> str:
        return """
You are a Prompt Engineer. Customize the original template for the specific task while preserving core functionality.

### Task:
#### User Instruction
{user_input_truncate_end_16384}
#### Description
{task_description}
#### Data Structure
{data_prompt}

### Original Template:
{target_prompt_template}

### Meta Instructions:
{meta_instructions}

### General Guidelines:
1. Only modify if the original template does not work for this task
2. Preserve core structure and variable placeholders (e.g., {<variable_name>})
3. Use truncation syntax when needed (e.g., {<variable_name>_truncate_end_2048})
4. Add relevant domain-specific knowledge

### CRITICAL: 
Your response must contain ONLY the prompt template text. Do NOT:
- Write code
- Provide examples
- Execute the prompt
- Add explanations
- Include commentary

Return the customized prompt template as plain text only.
"""

    def _build(self, **kwargs) -> str:
        """Build a prompt for the meta-prompting LLM.

        Args:
            **kwargs: Additional keyword arguments to customize the prompt building process
        """

        # We don't assert time_step here since meta-prompting might be used before the first step

        # Get the template to rewrite from the manager
        target_prompt_template = self.manager.target_prompt_instance.template
        meta_instructions = self.manager.target_prompt_instance.meta_instructions()

        # Get user input using the manager's standard properties
        try:
            user_input = self.manager.user_input
        except Exception:
            user_input = self.manager.initial_user_input

        # Render the prompt with additional variables
        additional_vars = {
            "target_prompt_template": target_prompt_template,
            "meta_instructions": meta_instructions,
            "user_input": user_input,
        }

        prompt = self.render(additional_vars)

        # Log the prompt for debugging if manager supports it
        if hasattr(self.manager, "save_and_log_states"):
            self.manager.save_and_log_states(
                content=prompt, save_name="meta_prompting_prompt.txt", per_iteration=True, add_uuid=False
            )

        return prompt

    def parse(self, response: Dict) -> str:
        """Parse the LLM's response to extract the rewritten template."""
        # Extract the rewritten template from the response
        rewritten_template = response.strip()

        # Save the response and rewritten template for debugging
        self.manager.save_and_log_states(
            content=response, save_name="meta_prompting_response.txt", per_iteration=True, add_uuid=False
        )
        self.manager.save_and_log_states(
            content=rewritten_template, save_name="rewritten_prompt_template.txt", per_iteration=True, add_uuid=False
        )

        return rewritten_template
