"""
Meta-prompting agent for dynamically rewriting prompts.

This module provides the MetaPromptingAgent class that can analyze the current task
and dynamically rewrite a specific prompt to better suit the requirements.
"""

import logging
from typing import Dict, Optional, Type

from ..prompts import MetaPromptingPrompt
from .base_agent import BaseAgent
from .utils import init_llm

logger = logging.getLogger(__name__)


class MetaPromptingAgent(BaseAgent):
    """
    Meta-prompting agent that dynamically rewrites a specific prompt based on the current task.
    
    This agent is designed to be instantiated for each prompt that needs rewriting,
    providing dedicated meta-prompting for each agent's prompt template.
    
    Agent Input:
        - Target prompt template to rewrite
        - Target prompt class for meta-instructions
        - Current task description and user input
        - Available template variables
        
    Agent Output:
        - Rewritten prompt template customized for the current task
    """
    
    def __init__(
        self, 
        config, 
        manager, 
        llm_config, 
        prompt_name,
        prompt_class,
        prompt_template=None,
        meta_prompt_template=None
    ):
        """
        Initialize the MetaPromptingAgent for a specific prompt.
        
        Args:
            config: Configuration object
            manager: Manager that provides state and variable values
            llm_config: Configuration for the language model
            prompt_name: Name of the prompt to rewrite (e.g., "python_coder")
            prompt_class: Class of the prompt (to access its meta_template)
            prompt_template: Template content to rewrite (default: prompt_class.default_template())
            meta_prompt_template: Optional custom template for the meta-prompting prompt
        """
        super().__init__(config=config, manager=manager)
        
        self.llm_config = llm_config
        self.prompt_name = prompt_name
        self.prompt_class = prompt_class
        
        # Get the template to rewrite if not provided
        if prompt_template is None and prompt_class is not None:
            # Create a temporary instance to get the default template
            temp_instance = prompt_class(llm_config=llm_config, manager=manager)
            prompt_template = temp_instance.default_template()
            
        # Store on the agent (not on manager to avoid conflicts with multiple agents)
        self.target_prompt_template = prompt_template
        
        # Initialize the meta-prompting prompt
        self.meta_prompt = MetaPromptingPrompt(
            llm_config=self.llm_config,
            manager=self.manager,
            template=meta_prompt_template
        )
        
        # Initialize the LLM lazily
        self.llm = None
        self._rewritten_template = None
        self._has_rewritten = False
    
    def __call__(self, user_input=None, data_prompt=None, task_description=None, target_prompt_template=None, agent_specific_instructions=None):
        """
        Generate a rewritten prompt template if not already done.
        
        Args:
            user_input: Optional user input to consider for meta-prompting
            data_prompt: Optional data information to consider for meta-prompting
            task_description: Optional task description to consider for meta-prompting
            target_prompt_template: Required template to be rewritten (if not provided earlier)
            agent_specific_instructions: Optional specific instructions for the agent
        
        Returns:
            Rewritten prompt template
        """
        # If already rewritten, return the cached version
        if self._has_rewritten:
            return self._rewritten_template
            
        self.manager.log_agent_start(f"MetaPromptingAgent: starting to analyze task and rewrite {self.prompt_name} template.")
        
        # Update target template if provided
        if target_prompt_template is not None:
            self.target_prompt_template = target_prompt_template
            
        # Ensure we have a target template to rewrite
        if not self.target_prompt_template:
            raise ValueError(f"No template provided to rewrite for {self.prompt_name}")
        
        # Prepare manager with necessary properties for the meta prompt
        self.manager.target_prompt_template = self.target_prompt_template
        self.manager.target_prompt_class = self.prompt_class
        
        # Set additional context information for meta-prompting
        if user_input is not None:
            self.manager.meta_user_input = user_input
            
        if data_prompt is not None:
            self.manager.meta_data_prompt = data_prompt
            
        if task_description is not None:
            self.manager.meta_task_description = task_description
            
        if agent_specific_instructions is not None:
            self.manager.meta_agent_specific_instructions = agent_specific_instructions
        
        # Build the meta-prompting prompt
        prompt = self.meta_prompt.build()
        
        # Initialize LLM if not already done
        if self.llm is None:
            self.llm = init_llm(
                llm_config=self.llm_config,
                agent_name=f"meta_prompting_{self.prompt_name}",
                multi_turn=self.llm_config.multi_turn
            )
        
        # Get response from LLM
        response = self.llm.assistant_chat(prompt)
        
        # Parse the response to get the rewritten template
        self._rewritten_template = self.meta_prompt.parse(response)
        self._has_rewritten = True
        
        # Save the rewritten template for debugging
        self.manager.save_and_log_states(
            content=self._rewritten_template,
            save_name=f"rewritten_{self.prompt_name}_template.txt",
            per_iteration=False,
            add_uuid=False
        )
        
        self.manager.log_agent_end(f"MetaPromptingAgent: finished rewriting {self.prompt_name} template.")
        
        # Clear temporary meta properties to avoid conflicts
        if hasattr(self.manager, 'meta_user_input'):
            delattr(self.manager, 'meta_user_input')
            
        if hasattr(self.manager, 'meta_data_prompt'):
            delattr(self.manager, 'meta_data_prompt')
            
        if hasattr(self.manager, 'meta_task_description'):
            delattr(self.manager, 'meta_task_description')
            
        if hasattr(self.manager, 'meta_agent_specific_instructions'):
            delattr(self.manager, 'meta_agent_specific_instructions')
        
        return self._rewritten_template
