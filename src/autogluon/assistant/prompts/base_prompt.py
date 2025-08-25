"""
Base prompt handling class.

This module provides the BasePrompt class which serves as the foundation
for all prompt types in the system.
"""

import logging
import importlib
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from .variable_provider import VariableProvider

# Import at module level to avoid circular import
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .meta_prompting_prompt import MetaPromptingPrompt

logger = logging.getLogger(__name__)


class BasePrompt(ABC):
    """Abstract base class for prompt handling"""

    def __init__(self, llm_config, manager, template=None):
        """
        Initialize prompt handler with configuration and optional template.

        Args:
            llm_config: Configuration for the language model
            manager: Manager that provides state and variable values
            template: Optional custom template. Can be:
                     - None: use default template
                     - A string path ending in .txt: load template from file
                     - A string: use as template directly
        """
        self.llm_config = llm_config
        self.manager = manager
        self.variable_provider = VariableProvider(manager)
            
        # State for meta-prompting
        self._original_template = template
        self._meta_prompted = False
        self._rewritten_template = None
        
        # Initialize the template (without meta-prompting, that will happen in build())
        self.set_template(template, apply_meta_prompting=False)

    def _load_template(self, template_str_or_path):
        if isinstance(template_str_or_path, str) and template_str_or_path.endswith(".txt"):
            try:
                logger.info(f"Loading template from file {template_str_or_path}")
                with open(template_str_or_path, "r") as f:
                    self.template = f.read()
            except Exception as e:
                logger.warning(f"Failed to load template from file {template_str_or_path}: {e}")
                self.template = self.default_template()
        else:
            self.template = template_str_or_path

        # Validate the template
        errors = self.variable_provider.validate_template(self.template)
        if errors:
            for error in errors:
                logger.warning(f"Template validation error: {error}")

    def set_template(self, template, apply_meta_prompting=False):
        """
        Set a new template, optionally applying meta-prompting to rewrite it.

        Args:
            template: Can be a file path ending in .txt or a template string
            apply_meta_prompting: Whether to apply meta-prompting (default: False)
                                  Typically, meta-prompting is applied during build() instead.
        """
        # First, get the base template
        if template is not None:
            self._load_template(template)
        elif self.llm_config.template is not None:
            self._load_template(self.llm_config.template)
        else:
            self.template = self.default_template()
            
        # Apply meta-prompting if explicitly requested
        # Note: We'll typically delay meta-prompting until build() is called
        if apply_meta_prompting:
            self.maybe_apply_meta_prompting()
            
    def maybe_apply_meta_prompting(self):
        """
        Apply meta-prompting if enabled and not already done.
        This is separated from set_template so it can be called at the right time,
        typically from build() when we have all the necessary context.
        """
        # Don't apply meta-prompting to the meta prompting prompt itself to avoid infinite recursion
        # Import here to avoid circular import
        from .meta_prompting_prompt import MetaPromptingPrompt
        if isinstance(self, MetaPromptingPrompt):
            return

        # Apply meta-prompting if enabled
        if (self.manager.enable_meta_prompting and not self._meta_prompted):
            self._apply_meta_prompting()
    
    def _apply_meta_prompting(self):
        """Apply meta-prompting to rewrite the current template."""
        logger.info(f"Applying meta-prompting to rewrite template for {self.__class__.__name__}")
        
        # Gather all the information needed for meta prompting
        # 1. Get the general meta prompting instruction from meta_prompting_prompt.py
        # This is already handled by the agent itself
        
        # 2. Get information asked in the instruction (template)
        # - task_description, user_input, data_prompt if available
        task_description = ""
        user_input = ""
        data_prompt = ""
        
        # Try to get task description
        if hasattr(self.manager, 'task_description'):
            task_description = self.manager.task_description
            
        # Try to get user input
        if hasattr(self.manager, 'time_step') and self.manager.time_step >= 0:
            try:
                user_input = self.manager.user_input
            except (AssertionError, IndexError):
                if hasattr(self.manager, 'user_inputs') and len(self.manager.user_inputs) > self.manager.time_step:
                    user_input = self.manager.user_inputs[self.manager.time_step]
                elif hasattr(self.manager, 'initial_user_input'):
                    user_input = self.manager.initial_user_input
        elif hasattr(self.manager, 'initial_user_input'):
            user_input = self.manager.initial_user_input
            
        # Try to get data prompt
        if hasattr(self.manager, 'data_prompt'):
            data_prompt = self.manager.data_prompt
            
        # 3. Get the target_prompt_template (current template) and agent_specific_instructions
        # Get agent-specific instructions from meta_template if available
        agent_specific_instructions = ""
        if hasattr(self.__class__, 'meta_template'):
            agent_specific_instructions = self.__class__.meta_template(self.__class__)
        
        # Store the original template and the current class on the manager
        # for the meta-prompting prompt to access
        self.manager.target_prompt_template = self.template
        self.manager.target_prompt_class = self.__class__
        
        # Use the existing meta-prompting agent with all required parameters
        self.manager.meta_prompting_agent.target_prompt_template = self.template
        self.manager.meta_prompting_agent.prompt_class = self.__class__
        rewritten_template = self.manager.meta_prompting_agent(
            user_input=user_input,
            data_prompt=data_prompt,
            task_description=task_description,
            target_prompt_template=self.template,
            agent_specific_instructions=agent_specific_instructions
        )

        # Save the rewritten template
        prompt_name = self.__class__.__name__
        self.manager.save_and_log_states(
            content=rewritten_template,
            save_name=f"rewritten_{prompt_name}_template.txt",
            per_iteration=False,
            add_uuid=False
        )
        
        # Update the template with the rewritten version
        self.template = rewritten_template
        self._meta_prompted = True
        self._rewritten_template = rewritten_template

        # Also store the rewritten template in the manager for reference
        prompt_class_name = self.__class__.__name__
        self.manager.rewritten_templates[prompt_class_name] = rewritten_template
        
        logger.info(f"Successfully applied meta-prompting to {self.__class__.__name__}")

    def _truncate_output_end(self, output: str, max_length: int) -> str:
        """Helper method to truncate output from the end if it exceeds max length"""
        if len(output) > max_length:
            truncated_text = f"\n[...TRUNCATED ({len(output) - max_length} characters)...]\n"
            return output[:max_length] + truncated_text
        return output

    def _truncate_output_mid(self, output: str, max_length: int) -> str:
        """Helper method to truncate output from the middle if it exceeds max length"""
        if len(output) > max_length:
            half_size = max_length // 2
            start_part = output[:half_size]
            end_part = output[-half_size:]
            truncated_text = f"\n[...TRUNCATED ({len(output) - max_length} characters)...]\n"
            return start_part + truncated_text + end_part
        return output

    def render(self, additional_vars: Optional[Dict[str, Any]] = None) -> str:
        """
        Render the prompt template with the current variable values.

        Args:
            additional_vars: Additional variables to use for this rendering only

        Returns:
            The rendered prompt
        """
        # If additional variables are provided, we need a temporary provider
        if additional_vars:
            # Create a subclass of VariableProvider that can handle the additional vars
            class TempProvider(VariableProvider):
                def __init__(self, parent_provider, additional_vars):
                    self.parent_provider = parent_provider
                    self.additional_vars = additional_vars
                    # Keep a reference to the manager for method calls
                    self.manager = parent_provider.manager

                def get_value(self, var_name):
                    if var_name in self.additional_vars:
                        return self.additional_vars[var_name]
                    return self.parent_provider.get_value(var_name)

            temp_provider = TempProvider(self.variable_provider, additional_vars)
            rendered = temp_provider.render_template(self.template)
        else:
            rendered = self.variable_provider.render_template(self.template)

        # Add format instructions if configured
        if hasattr(self.llm_config, "add_coding_format_instruction") and self.llm_config.add_coding_format_instruction:
            if hasattr(self, "get_format_instruction"):
                format_instruction = self.get_format_instruction()
                rendered = f"{rendered}\n\n{format_instruction}"

        return rendered

    def build(self) -> str:
        """
        Build the prompt string.
        
        This method applies meta-prompting if enabled, then calls the _build method
        which should be implemented by subclasses.
        """
        # Apply meta-prompting if appropriate - this ensures we have the latest context
        self.maybe_apply_meta_prompting()
        
        # Call the template method that subclasses should override
        return self._build()
    
    def _build(self) -> str:
        """
        Template method for building the prompt string.
        
        Subclasses should override this method instead of build().
        """
        raise NotImplementedError("Subclasses must implement _build()")

    @abstractmethod
    def parse(self, response: Dict) -> any:
        """Parse the LLM response"""
        pass

    @abstractmethod
    def default_template(self) -> str:
        """Default prompt template"""
        pass
        
    def meta_template(self) -> str:
        """
        Template specifically for meta-prompting.
        
        This template provides instructions on how to rewrite this prompt
        to better suit a specific task. Subclasses should override this
        method to provide prompt-specific guidance.
        """
        raise NotImplementedError("Subclasses must implement meta_template()")
