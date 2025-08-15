"""
Variable registry for prompt templates.

This module provides a centralized registry for prompt template variables,
allowing for consistent naming and access across different prompt classes.
"""

import logging
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class VariableDefinition:
    """Defines a template variable with metadata."""

    def __init__(
        self,
        name: str,
        description: str,
        aliases: Optional[List[str]] = None,
        deprecated_aliases: Optional[List[str]] = None,
    ):
        """
        Initialize variable definition.

        Args:
            name: The canonical variable name
            description: Description of what this variable represents
            aliases: Alternative names that can be used for this variable
            deprecated_aliases: Older names that should be avoided
        """
        self.name = name
        self.description = description
        self.aliases = aliases or []
        self.deprecated_aliases = deprecated_aliases or []

    def get_all_names(self) -> Set[str]:
        """Get all possible names for this variable (canonical + aliases)."""
        return {self.name, *self.aliases, *self.deprecated_aliases}


class VariableRegistry:
    """Registry for prompt template variables."""

    def __init__(self):
        # Map of canonical variable name to VariableDefinition
        self.variables: Dict[str, VariableDefinition] = {}

        # Map of variable name or alias to canonical name
        self.name_map: Dict[str, str] = {}

        # Initialize the registry with standard variables
        self._initialize_registry()

    def _initialize_registry(self):
        """Initialize the registry with standard variables."""
        # User-related variables
        self.register(
            VariableDefinition(
                name="user_input",
                description="User instructions and requirements",
                aliases=["user_prompt"],
            )
        )

        # Task-related variables
        self.register(
            VariableDefinition(
                name="task_description",
                description="Description of the ML task to be performed",
            )
        )

        # Data-related variables
        self.register(
            VariableDefinition(
                name="data_prompt",
                description="Information about data structure and files",
            )
        )

        # Output-related variables
        self.register(
            VariableDefinition(
                name="per_iteration_output_folder",
                description="Output directory for generated code at current iteration",
            )
        )

        # Error-related variables
        self.register(
            VariableDefinition(
                name="previous_error_message",
                description="Error message in the previous iteration",
            )
        )

        # Error-related variables
        self.register(
            VariableDefinition(
                name="all_error_analyses",
                description="Error analysis in all completed iterations",
            )
        )

        # Code-related variables
        self.register(VariableDefinition(name="python_code", description="Python code of current iteration"))

        # Code-related variables
        self.register(VariableDefinition(name="previous_python_code", description="Python code of previous iteration"))

        self.register(
            VariableDefinition(
                name="python_file_path",
                description="Path to the Python file to execute",
            )
        )

        self.register(
            VariableDefinition(
                name="bash_script",
                description="Bash script of current iter",
            )
        )

        self.register(
            VariableDefinition(
                name="previous_bash_script",
                description="Bash script of previous iter",
            )
        )

        # Environment-related variables
        self.register(
            VariableDefinition(
                name="environment_prompt",
                description="System environment information",
            )
        )

        # Tool-related variables
        self.register(
            VariableDefinition(
                name="selected_tool",
                description="The chosen ML library/framework",
            )
        )

        self.register(
            VariableDefinition(
                name="tool_prompt",
                description="Information about selected tools",
                aliases=["tools_info"],
            )
        )

        # Tutorial-related variables
        self.register(
            VariableDefinition(
                name="tutorial_prompt",
                description="Relevant tutorial information of current iteration",
            )
        )

        # Tutorial-related variables
        self.register(
            VariableDefinition(
                name="previous_tutorial_prompt",
                description="Relevant tutorial information in previous iteration",
            )
        )

        # Execution-related variables
        self.register(
            VariableDefinition(
                name="stdout",
                description="Standard output from code execution",
            )
        )

        self.register(
            VariableDefinition(
                name="stderr",
                description="Standard error from code execution",
            )
        )

        # Validation-related variables
        self.register(
            VariableDefinition(
                name="validation_prompt",
                description="Validation criteria",
            )
        )

        # Best code related
        self.register(
            VariableDefinition(
                name="best_code_prompt",
                description="Examples of high-quality code",
            )
        )

        # File-related variables
        self.register(
            VariableDefinition(
                name="file_path",
                description="Path to the file being read",
            )
        )

        self.register(
            VariableDefinition(
                name="file_size_mb",
                description="Size of the file in MB",
            )
        )

        self.register(
            VariableDefinition(
                name="max_chars",
                description="Maximum characters to read",
            )
        )

        # Description file related
        self.register(
            VariableDefinition(
                name="description_file_contents",
                description="Contents of identified description files",
            )
        )

        # Reranker related
        self.register(
            VariableDefinition(
                name="max_num_tutorials",
                description="Maximum number of tutorials to select",
            )
        )

    def register(self, var_def: VariableDefinition):
        """Register a variable definition."""
        self.variables[var_def.name] = var_def

        # Add all names to the name map
        self.name_map[var_def.name] = var_def.name
        for alias in var_def.aliases:
            self.name_map[alias] = var_def.name
        for deprecated in var_def.deprecated_aliases:
            self.name_map[deprecated] = var_def.name

    def get_canonical_name(self, name: str) -> str:
        """
        Get the canonical name for a variable name or alias.

        Args:
            name: The variable name or alias

        Returns:
            The canonical variable name

        Raises:
            ValueError: If the name is not registered
        """
        if name not in self.name_map:
            raise ValueError(f"Unknown variable name: {name}")
        return self.name_map[name]

    def get_all_variables(self) -> Dict[str, VariableDefinition]:
        """Get all registered variables."""
        return self.variables

    def get_variable_info(self, name: str) -> VariableDefinition:
        """
        Get information about a variable.

        Args:
            name: The variable name or alias

        Returns:
            Variable definition

        Raises:
            ValueError: If the name is not registered
        """
        canonical_name = self.get_canonical_name(name)
        return self.variables[canonical_name]


# Singleton instance of the registry
registry = VariableRegistry()
