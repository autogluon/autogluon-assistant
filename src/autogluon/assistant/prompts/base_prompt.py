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
        self.template = None
            
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
        
        # Meta prompting will use the standard manager variables
        # No need to gather variables separately, as they'll be accessed directly from the manager
        # The meta-prompting prompt class knows how to access these variables via the manager
        # Store the original template and the current class on the manager
        # for the meta-prompting prompt to access
        self.manager.target_prompt_class = self
        
        # Use the existing meta-prompting agent with all required parameters
        prompt_name = self.__class__.__name__
        rewritten_template = self.manager.meta_prompting_agent(
            target_prompt_instance=self
        )

        # The meta_prompting_agent already saves the rewritten template, no need to save again
        
        # Update the template with the rewritten version
        self.template = rewritten_template
        self._meta_prompted = True
        self._rewritten_template = rewritten_template
        
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

    def build(self, **kwargs) -> str:
        """
        Build the prompt string.
        
        This method applies meta-prompting if enabled, then calls the _build method
        which should be implemented by subclasses.
        
        Args:
            **kwargs: Additional keyword arguments to pass to the _build method.
                     These can be used to customize the prompt building process.
        
        Returns:
            str: The built prompt string
        """
        # Apply meta-prompting if appropriate - this ensures we have the latest context
        self.maybe_apply_meta_prompting()
        
        # Call the template method that subclasses should override, passing all kwargs
        return self._build(**kwargs)
    
    def _build(self, **kwargs) -> str:
        """
        Template method for building the prompt string.
        
        Subclasses should override this method instead of build().
        
        Args:
            **kwargs: Additional keyword arguments to customize the prompt building process.
                     These are passed from the build() method.
        
        Returns:
            str: The built prompt string
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
        
    @classmethod
    def meta_instructions(cls) -> str:
        """
        Template specifically for meta-prompting.
        
        This template provides instructions on how to rewrite this prompt
        to better suit a specific task. Subclasses should override this
        method to provide prompt-specific guidance.
        """
        raise NotImplementedError("Subclasses must implement meta_instructions()")
