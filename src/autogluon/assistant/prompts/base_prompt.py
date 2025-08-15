"""
Base prompt handling class.

This module provides the BasePrompt class which serves as the foundation
for all prompt types in the system.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from .variable_provider import VariableProvider

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
        self.set_template(template)

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

    def set_template(self, template):
        """
        Set a new template.

        Args:
            template: Can be a file path ending in .txt or a template string
        """
        if template is not None:
            self._load_template(template)
        elif self.llm_config.template is not None:
            self._load_template(self.llm_config.template)
        else:
            self.template = self.default_template()

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

    @abstractmethod
    def build(self) -> str:
        """Build the prompt string"""
        pass

    @abstractmethod
    def parse(self, response: Dict) -> any:
        """Parse the LLM response"""
        pass

    @abstractmethod
    def default_template(self) -> str:
        """Default prompt template"""
        pass
