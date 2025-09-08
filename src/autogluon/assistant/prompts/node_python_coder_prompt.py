"""
Python code generation prompt for NodeManager.

This module provides a variant of the PythonCoderPrompt class for generating Python code
when using the NodeManager with MCTS.
"""

import logging

from .python_coder_prompt import PythonCoderPrompt

logger = logging.getLogger(__name__)


class NodePythonCoderPrompt(PythonCoderPrompt):
    """Handles prompts for Python code generation with NodeManager"""

    def _generate_code_improvement_prompt(self) -> str:
        """Generate prompt section about best/successful previous code, adapted for NodeManager."""
        if self.manager.time_step == 0:
            return ""  # No previous code on first iteration

        code_improvement_prompt = []

        # Check if we're in an evolve node with a parent node
        current_node = self.manager.current_node
        if current_node and current_node.stage == "evolve" and current_node.parent:
            # For evolve nodes, provide the parent's code as reference
            parent_code = current_node.parent.python_code
            parent_score = current_node.parent.validation_score

            code_improvement_prompt.append("### Previous Successful Code")
            if parent_score is not None:
                code_improvement_prompt.append(
                    f"The following code achieved a validation score of {parent_score:.4f}:"
                )
            else:
                code_improvement_prompt.append("The following code executed successfully:")

            code_improvement_prompt.append("```python")
            code_improvement_prompt.append(parent_code)
            code_improvement_prompt.append("```")
            code_improvement_prompt.append("")
            code_improvement_prompt.append(
                "Please prioritize model architecture improvements and training optimization to enhance performance. Feature engineering may also be applied but with lower priority."
            )

            if (
                hasattr(self.manager.config, "optimize_system_resources")
                and self.manager.config.optimize_system_resources
            ):
                code_improvement_prompt.append(self._generate_system_resources_prompt())

        # Check if we have a best node
        elif self.manager.best_node and self.manager.best_node != current_node:
            # Use the best node's code
            best_code = self.manager.best_node.python_code
            best_score = self.manager.best_validation_score

            code_improvement_prompt.append("### Previous Best Code")
            code_improvement_prompt.append(
                f"The following code achieved the best validation score so far ({best_score:.4f}):"
            )
            code_improvement_prompt.append("```python")
            code_improvement_prompt.append(best_code)
            code_improvement_prompt.append("```")
            code_improvement_prompt.append("")
            code_improvement_prompt.append(
                "Please prioritize model architecture improvements and training optimization to enhance performance. Feature engineering may also be applied but with lower priority."
            )

            if (
                hasattr(self.manager.config, "optimize_system_resources")
                and self.manager.config.optimize_system_resources
            ):
                code_improvement_prompt.append(self._generate_system_resources_prompt())

        # Check if we have a last successful node (different from best node)
        elif self.manager.last_successful_node and self.manager.last_successful_node != current_node:
            successful_code = self.manager.last_successful_node.python_code

            code_improvement_prompt.append("### Previous Successful Code")
            code_improvement_prompt.append("The following code executed successfully:")
            code_improvement_prompt.append("```python")
            code_improvement_prompt.append(successful_code)
            code_improvement_prompt.append("```")
            code_improvement_prompt.append("")
            code_improvement_prompt.append(
                "Please prioritize model architecture improvements and training optimization to enhance performance. Feature engineering may also be applied but with lower priority."
            )

            if (
                hasattr(self.manager.config, "optimize_system_resources")
                and self.manager.config.optimize_system_resources
            ):
                code_improvement_prompt.append(self._generate_system_resources_prompt())

        # For debug nodes, always show the parent's code that needs debugging
        elif current_node and current_node.stage == "debug" and current_node.parent:
            parent_code = current_node.parent.python_code

            code_improvement_prompt.append("### Code To Debug")
            code_improvement_prompt.append("The following code has errors that need to be fixed:")
            code_improvement_prompt.append("```python")
            code_improvement_prompt.append(parent_code)
            code_improvement_prompt.append("```")
            code_improvement_prompt.append("")
            code_improvement_prompt.append(
                "Please identify and fix the errors in the code above. Make minimal changes necessary to fix the issues."
            )

        # Do nothing if there's no successful code
        else:
            code_improvement_prompt = []

        return "\n".join(code_improvement_prompt)
