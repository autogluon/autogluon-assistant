"""
Variable registry for prompt templates.

This module provides a centralized registry for prompt template variables,
allowing for consistent naming and access across different prompt classes.
"""

import logging
from typing import Dict, Any, Set, List, Optional

logger = logging.getLogger(__name__)


class VariableDefinition:
    """Defines a template variable with metadata."""
    
    def __init__(
        self, 
        name: str, 
        description: str, 
        aliases: Optional[List[str]] = None,
        deprecated_aliases: Optional[List[str]] = None
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
                name="output_folder",
                description="Output directory for generated code",
            )
        )
        
        # Error-related variables
        self.register(
            VariableDefinition(
                name="error_message",
                description="Detailed error message",
                deprecated_aliases=["error_analysis"],
            )
        )
        
        # Code-related variables
        self.register(
            VariableDefinition(
                name="python_code",
                description="Python code content",
                aliases=["current_python"],
            )
        )
        
        self.register(
            VariableDefinition(
                name="python_file_path",
                description="Path to the Python file to execute",
            )
        )
        
        self.register(
            VariableDefinition(
                name="bash_script",
                description="Bash script that was executed",
                aliases=["previous_bash"],
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
                description="Relevant tutorial information",
                aliases=["tutorials_info"],
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
        
        # Previous error specific variables
        self.register(
            VariableDefinition(
                name="all_previous_error_prompts",
                description="All error prompts from previous iterations",
                aliases=["previous_error_prompt", "error_prompt"],
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
    
    def get_variables_by_category(self) -> Dict[str, List[VariableDefinition]]:
        """
        Get variables grouped by category (based on naming conventions).
        
        Returns:
            Dictionary of category -> list of variables
        """
        categories = {}
        
        # Define categories based on naming patterns
        for var_name, var_def in self.variables.items():
            if "_prompt" in var_name:
                category = "prompts"
            elif "_code" in var_name or var_name.endswith("_script"):
                category = "code"
            elif "error" in var_name:
                category = "errors"
            elif "file" in var_name:
                category = "files"
            elif "tool" in var_name:
                category = "tools"
            elif var_name in ["stdout", "stderr"]:
                category = "execution"
            else:
                category = "general"
                
            if category not in categories:
                categories[category] = []
            categories[category].append(var_def)
            
        return categories


# Singleton instance of the registry
registry = VariableRegistry()