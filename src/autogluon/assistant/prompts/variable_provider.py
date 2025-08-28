"""
Variable provider for prompt templates.

This module provides a system for retrieving variable values from the manager
and rendering templates with those values.
"""

import logging
import re
from typing import Any, Dict, List, Set, Tuple

from .variables import registry

logger = logging.getLogger(__name__)


class VariableProvider:
    """Provides variable values for prompt templates."""

    def __init__(self, manager):
        """
        Initialize the variable provider.

        Args:
            manager: The Manager instance that holds state and provides values
        """
        self.manager = manager

    def get_value(self, var_name: str) -> Any:
        """
        Get the value for a variable.

        Args:
            var_name: The variable name (can be an alias)

        Returns:
            The variable value

        Raises:
            ValueError: If the variable cannot be retrieved
        """
        try:
            canonical_name = registry.get_canonical_name(var_name)
        except ValueError:
            logger.warning(f"Unknown variable requested: {var_name}")
            return f"{{UNKNOWN_VARIABLE:{var_name}}}"

        # Handle deprecated aliases with a warning
        var_info = registry.get_variable_info(canonical_name)
        if var_name in var_info.deprecated_aliases:
            logger.warning(f"Using deprecated variable name '{var_name}'. " f"Please use '{canonical_name}' instead.")

        # Get the value using the appropriate method based on the canonical name
        try:
            # First check if the manager has a property with this name
            if hasattr(self.manager, canonical_name):
                return getattr(self.manager, canonical_name)

            # If not, use a mapping of special cases
            # This is a transitional approach until all variables are
            # consistently named in the manager
            special_cases = {
                "user_input": lambda: self.manager.user_input,
                "task_description": lambda: self.manager.task_description,
                "data_prompt": lambda: self.manager.data_prompt,
                "per_iteration_output_folder": lambda: self.manager.per_iteration_output_folder,
                "python_code": lambda: self.manager.python_code,
                "previous_python_code": lambda: self.manager.previous_python_code,
                "python_file_path": lambda: self.manager.python_file_path,
                "bash_script": lambda: self.manager.bash_script,
                "previous_bash_script": lambda: self.manager.previous_bash_script,
                "previous_error_message": lambda: self.manager.previous_error_message,
                "all_error_analyses": lambda: self.manager.all_previous_error_analyses,
                "tutorial_prompt": lambda: self.manager.tutorial_prompt,
                "previous_tutorial_prompt": lambda: self.manager.previous_tutorial_prompt,
                "selected_tool": lambda: self.manager.selected_tool,
                "tool_prompt": lambda: self.manager.tool_prompt,
                # Add more mappings as needed
            }

            if canonical_name in special_cases:
                return special_cases[canonical_name]()

            # If we reach here, we don't know how to get this value
            logger.warning(f"No method to retrieve variable: {canonical_name}")
            return f"{{UNAVAILABLE_VARIABLE:{canonical_name}}}"

        except Exception as e:
            logger.warning(f"Error getting value for {canonical_name}: {e}")
            return f"{{ERROR_VARIABLE:{canonical_name}}}"

    def get_all_available_variables(self) -> Dict[str, Any]:
        """
        Get all available variables and their current values.

        Returns:
            Dict of variable names to values
        """
        result = {}
        for var_name in registry.get_all_variables().keys():
            try:
                result[var_name] = self.get_value(var_name)
            except:
                # Skip variables that can't be retrieved
                pass
        return result

    def extract_variables_from_template(self, template: str) -> Set[str]:
        """
        Extract all variable names used in a template.

        Args:
            template: The template string

        Returns:
            Set of variable names
        """
        # This regex finds all {variable_name} patterns
        # but ignores escaped braces like \{not_a_variable\}
        # and excludes any {} that contain < or > characters
        pattern = r"(?<!\\){([^{}<>]+)}"
        return set(re.findall(pattern, template))

    def validate_template(self, template: str) -> List[str]:
        """
        Validate a template by checking if all variables exist.

        Args:
            template: The template string

        Returns:
            List of validation errors, empty if valid
        """
        errors = []
        template_vars = self.extract_variables_from_template(template)

        for var in template_vars:
            try:
                registry.get_canonical_name(var)
            except ValueError:
                errors.append(f"Unknown variable: {var}")

        return errors

    def _parse_variable_with_truncation(self, var_name: str) -> Tuple[str, str, int]:
        """
        Parse variable name to extract truncation directive if present.
        Format: variable_name_truncate_mode_length
        Examples:
        - xxx_truncate_mid_2048 -> truncate in middle to 2048 chars
        - xxx_truncate_start_4096 -> truncate from start to 4096 chars
        - xxx_truncate_end_1024 -> truncate from end to 1024 chars

        Args:
            var_name: The variable name to parse

        Returns:
            Tuple of (base_var_name, truncate_mode, max_length)
            - truncate_mode is one of 'start', 'mid', 'end' or None
            - max_length is an integer or None
        """
        # Check for truncation directive pattern
        pattern = r"^(.+)_truncate_(start|mid|end)_(\d+)$"
        match = re.match(pattern, var_name)

        if match:
            base_var_name = match.group(1)
            truncate_mode = match.group(2)
            max_length = int(match.group(3))
            return base_var_name, truncate_mode, max_length

        return var_name, None, None

    def _truncate_value(self, value: str, truncate_mode: str, max_length: int) -> str:
        """
        Truncate a string value according to the specified mode and length.

        Args:
            value: The string value to truncate
            truncate_mode: One of 'start', 'mid', 'end'
            max_length: Maximum length to truncate to

        Returns:
            Truncated string
        """
        if not isinstance(value, str):
            value = str(value)

        if len(value) <= max_length:
            return value

        truncated_text = f"\n[...TRUNCATED ({len(value) - max_length} characters)...]\n"

        if truncate_mode == "start":
            return truncated_text + value[-max_length:]
        elif truncate_mode == "end":
            return value[:max_length] + truncated_text
        elif truncate_mode == "mid":
            half_size = max_length // 2
            start_part = value[:half_size]
            end_part = value[-half_size:]
            return start_part + truncated_text + end_part
        else:
            logger.warning(f"Unknown truncation mode: {truncate_mode}")
            return value

    def render_template(self, template: str) -> str:
        """
        Render a template by replacing variables with their values.
        Supports truncation syntax: {variable_name_truncate_mode_length}

        Args:
            template: The template string

        Returns:
            The rendered template
        """
        if not template:
            return ""

        template_vars = self.extract_variables_from_template(template)
        rendered = template

        for var in template_vars:
            try:
                # Parse variable name for possible truncation directive
                base_var_name, truncate_mode, max_length = self._parse_variable_with_truncation(var)

                # Get the value for the base variable name
                value = self.get_value(base_var_name)

                # Apply truncation if specified
                if truncate_mode and max_length and isinstance(value, str):
                    value = self._truncate_value(value, truncate_mode, max_length)

                # Replace {var} with the actual value
                rendered = rendered.replace(f"{{{var}}}", str(value or ""))
            except Exception as e:
                logger.warning(f"Error rendering variable {var}: {e}")
                # Leave the variable in place if there's an error

        return rendered
