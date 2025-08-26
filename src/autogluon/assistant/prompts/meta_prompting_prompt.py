"""
Meta-prompting prompt class to dynamically rewrite other prompts based on current task.

This module provides the MetaPromptingPrompt class which can analyze the current
task and variables to generate customized prompts for other agents.
"""

import logging
from typing import Dict, Optional

from .base_prompt import BasePrompt
from .utils import extract_text_between

logger = logging.getLogger(__name__)


class MetaPromptingPrompt(BasePrompt):
    """Handles meta-prompting for customizing other agent prompts"""

    def default_template(self) -> str:
        return """
You are a Meta Prompt Engineer tasked with customizing a template for an AI assistant. Your job is to rewrite the template to better suit a specific task while preserving its core functionality.

### Current Context:
Task Description: {task_description}
Data Information: {data_prompt}
User Request: {user_input_truncate_end_8192}

### Original Template:
{target_prompt_template}

### Domain-Specific Instructions:
{meta_instructions}

### Available Variables:
{available_variables}

### General Guidelines:
1. Maintain the original template's purpose and core structure
2. Preserve necessary variable placeholders (e.g., {variable_name})
3. Add domain-specific knowledge relevant to the current task
4. Keep instructions precise, concise, and specific to the task
5. You may use variable truncation syntax (e.g., {variable_name_truncate_end_2048}) where appropriate

Your response must ONLY contain the rewritten template with no additional explanations or commentary.
"""

    def _build(self) -> str:
        """Build a prompt for the meta-prompting LLM."""
        # We don't assert time_step here since meta-prompting might be used before the first step
        
        # Get available variables and their values for context
        available_variables = self._get_available_variables_description()
        variable_values = self._get_variable_values_context()
        
        # Get the template to rewrite from the manager
        target_prompt_template = ""
        meta_instructions = ""
        target_prompt_class = None
        
        if hasattr(self.manager, 'target_prompt_template'):
            target_prompt_template = self.manager.target_prompt_template
        
        # Get meta instructions from the target prompt class or specific instructions
        if hasattr(self.manager, 'meta_agent_specific_instructions') and self.manager.meta_agent_specific_instructions:
            meta_instructions = self.manager.meta_agent_specific_instructions
        elif hasattr(self.manager, 'target_prompt_class') and self.manager.target_prompt_class is not None:
            target_prompt_class = self.manager.target_prompt_class
            if hasattr(target_prompt_class, 'meta_template'):
                meta_instructions = target_prompt_class.meta_template(self.manager.target_prompt_class)
        
        # Get task-specific context for better meta-prompting, with priority to meta-specific inputs
        # Check for meta-specific task description first
        if hasattr(self.manager, 'meta_task_description'):
            task_description = self.manager.meta_task_description
        else:
            task_description = getattr(self.manager, 'task_description', '')
        
        # Check for meta-specific data prompt
        if hasattr(self.manager, 'meta_data_prompt'):
            data_prompt = self.manager.meta_data_prompt
        else:
            data_prompt = getattr(self.manager, 'data_prompt', '')
        
        # Get agent-specific instructions if available
        agent_specific_instructions = getattr(self.manager, 'meta_agent_specific_instructions', '')
        
        # Get user input with priority to meta-specific input
        user_input = ""
        if hasattr(self.manager, 'meta_user_input'):
            # Use meta-specific user input first if available
            user_input = self.manager.meta_user_input
        elif hasattr(self.manager, 'time_step') and self.manager.time_step >= 0:
            # Try to get user_input via property accessor
            try:
                user_input = self.manager.user_input
            except (AssertionError, IndexError):
                # If not available via property, try direct attribute access
                if hasattr(self.manager, 'user_inputs') and len(self.manager.user_inputs) > self.manager.time_step:
                    user_input = self.manager.user_inputs[self.manager.time_step]
                elif hasattr(self.manager, 'initial_user_input'):
                    # Fall back to initial user input if available
                    user_input = self.manager.initial_user_input
        elif hasattr(self.manager, 'initial_user_input'):
            # Use initial user input if time_step not available
            user_input = self.manager.initial_user_input
            
        # Render the prompt with additional variables
        additional_vars = {
            "available_variables": available_variables,
            "variable_values": variable_values,
            "target_prompt_template": target_prompt_template,
            "meta_instructions": meta_instructions,
            "task_description": task_description,
            "data_prompt": data_prompt,
            "user_input": user_input,
            "agent_specific_instructions": agent_specific_instructions
        }
        
        prompt = self.render(additional_vars)
        
        # Log the prompt for debugging if manager supports it
        if hasattr(self.manager, 'save_and_log_states'):
            self.manager.save_and_log_states(
                content=prompt, save_name="meta_prompting_prompt.txt", per_iteration=True, add_uuid=False
            )
        
        return prompt

    def parse(self, response: Dict) -> str:
        """Parse the LLM's response to extract the rewritten template."""
        # Extract the rewritten template from the response
        rewritten_template = response.get("content", "").strip()
        
        # Save the response and rewritten template for debugging
        self.manager.save_and_log_states(
            content=response, save_name="meta_prompting_response.txt", per_iteration=True, add_uuid=False
        )
        self.manager.save_and_log_states(
            content=rewritten_template, save_name="rewritten_prompt_template.txt", per_iteration=True, add_uuid=False
        )
        
        return rewritten_template

    def _get_available_variables_description(self) -> str:
        """Get a description of all available template variables."""
        descriptions = []
        
        # Add descriptions for common variables from the registry
        from .variables import registry
        for name, var_def in registry.get_all_variables().items():
            descriptions.append(f"- {name}: {var_def.description}")
            
            # Add aliases if they exist
            if var_def.aliases:
                descriptions.append(f"  Aliases: {', '.join(var_def.aliases)}")
                
        return "\n".join(descriptions)
    
    def _get_variable_values_context(self) -> str:
        """Get the current values of important variables for context."""
        # Get a subset of variables that are most relevant for context
        context_vars = [
            "task_description", 
            "data_prompt", 
            "selected_tool",
            "tool_prompt"
        ]
        
        result = []
        for var_name in context_vars:
            try:
                value = self.variable_provider.get_value(var_name)
                # Truncate long values for readability
                if isinstance(value, str) and len(value) > 500:
                    value = value[:250] + "..." + value[-250:]
                result.append(f"### {var_name}:\n{value}\n")
            except:
                # Skip variables that can't be retrieved
                pass
                
        return "\n".join(result)