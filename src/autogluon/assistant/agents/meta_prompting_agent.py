"""
Meta-prompting agent for dynamically rewriting prompts.

This module provides the MetaPromptingAgent class that can analyze the current task
and dynamically rewrite a specific prompt to better suit the requirements.
"""

import logging

from ..prompts import MetaPromptingPrompt
from .base_agent import BaseAgent
from .utils import init_llm

logger = logging.getLogger(__name__)


class MetaPromptingAgent(BaseAgent):
    """
    Meta-prompting agent that dynamically rewrites prompts based on the current task.

    This agent is designed to be instantiated once and used for all prompts that need rewriting,
    handling any prompt template provided during the call.

    Agent Input:
        - Target prompt template to rewrite
        - Target prompt class for meta-instructions
        - Current task description and user input
        - Available template variables

    Agent Output:
        - Rewritten prompt template customized for the current task
    """

    def __init__(self, config, manager, llm_config, meta_prompt_template=None):
        """
        Initialize the MetaPromptingAgent for all prompts.

        Args:
            config: Configuration object
            manager: Manager that provides state and variable values
            llm_config: Configuration for the language model
            meta_prompt_template: Optional custom template for the meta-prompting prompt
        """
        super().__init__(config=config, manager=manager)

        self.llm_config = llm_config

        # Initialize the meta-prompting prompt
        self.meta_prompt = MetaPromptingPrompt(
            llm_config=self.llm_config, manager=self.manager, template=meta_prompt_template
        )

        # Initialize the LLM lazily
        self.llm = None
        # Store rewritten templates in a dictionary keyed by prompt class name
        self._rewritten_templates = {}

    def __call__(self, target_prompt_instance, force_rewrite=False):
        """
        Generate a rewritten prompt template for the specified prompt class.

        Args:
            target_prompt_instance: Instance of the prompt (to access its meta_instructions)
            force_rewrite: If True, rewrite the template even if previously rewritten

        Returns:
            Rewritten prompt template
        """
        assert self.manager.target_prompt_instance is None, f"{self.manager.target_prompt_instance.__class__}"
        self.manager.target_prompt_instance = target_prompt_instance
        # Generate a key to identify this prompt in our cache
        prompt_name = target_prompt_instance.__class__.__name__

        # If already rewritten and not forcing a rewrite, return the cached version
        if not force_rewrite and prompt_name in self._rewritten_templates:
            self.manager.target_prompt_instance = None
            return self._rewritten_templates[prompt_name]

        self.manager.log_agent_start(
            f"MetaPromptingAgent: starting to analyze task and rewrite {prompt_name} template."
        )

        # Build the meta-prompting prompt
        prompt = self.meta_prompt.build()

        # Initialize LLM if not already done
        if self.llm is None:
            self.llm = init_llm(
                llm_config=self.llm_config, agent_name="meta_prompting", multi_turn=self.llm_config.multi_turn
            )

        # Get response from LLM
        response = self.llm.assistant_chat(prompt)

        # Parse the response to get the rewritten template
        rewritten_template = self.meta_prompt.parse(response)

        # Cache the rewritten template
        self._rewritten_templates[prompt_name] = rewritten_template

        # Save the rewritten template for debugging
        self.manager.save_and_log_states(
            content=rewritten_template,
            save_name=f"rewritten_{prompt_name}_template.txt",
            per_iteration=False,
            add_uuid=False,
        )

        self.manager.log_agent_end(f"MetaPromptingAgent: finished rewriting {prompt_name} template.")

        self.manager.target_prompt_instance = None

        return rewritten_template
