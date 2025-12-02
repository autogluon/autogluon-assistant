"""
Chat prompt for conversational interactions.

This module provides the ChatPrompt class for generating responses to user
questions in a conversational manner, with optional data context.
"""

import logging
from typing import Dict

from .base_prompt import BasePrompt

logger = logging.getLogger(__name__)


class ChatPrompt(BasePrompt):
    """Handles prompts for chat-based question answering"""

    @classmethod
    def meta_instructions(cls) -> str:
        """
        Returns specific instructions for meta-prompting the chat template.
        """
        return """
This prompt generates helpful responses to user questions about data analysis, machine learning,
and related topics. The agent should be informative, accurate, and helpful without executing code.
Make sure to PRESERVE the variables in the original template.
"""

    def default_template(self) -> str:
        return """You are an AutoML Assistant designed to help users understand data analysis, machine learning techniques, and best practices. Your role is to answer questions, provide guidance, and explain concepts clearly.

### Instructions
1. Answer the user's question thoroughly and accurately
2. If relevant, reference the data context or tutorials provided
3. Provide practical examples or code snippets when helpful (but do not execute them)
4. If the question is unclear, ask for clarification
5. Be conversational and helpful
6. If you don't know something, admit it rather than guessing
7. **Format your response in markdown** with proper headers, code blocks, lists, and emphasis where appropriate

{data_context_prompt}

{tutorial_prompt}

---

User Question: {user_message}

Please provide your response in well-formatted markdown."""

    def get_format_instruction(self) -> str:
        """Get the format instruction to append to the prompt."""
        return "Please provide a clear, well-structured response. Use markdown formatting for code examples."

    def _build(self, **kwargs) -> str:
        """Build a prompt for the LLM to generate a chat response.

        Args:
            **kwargs: Additional keyword arguments to customize the prompt building process
        """
        # Get the current user message
        user_message = getattr(self.manager, "current_user_message", "")

        # Generate data context prompt only if not yet presented
        data_context_prompt = self._generate_data_context_prompt()

        # Tutorial prompt is already set by the manager
        tutorial_section = self._generate_tutorial_section()

        # Render the prompt using the variable provider with additional variables
        rendered = self.render(
            additional_vars={
                "data_context_prompt": data_context_prompt,
                "tutorial_prompt": tutorial_section,
                "user_message": user_message,
            }
        )

        return rendered

    def _generate_data_context_prompt(self) -> str:
        """
        Generate a prompt section with data context if available and not yet presented.

        Returns:
            Data context prompt section (empty if already presented)
        """
        # Only include data context if it hasn't been presented yet
        if hasattr(self.manager, "data_context_presented") and self.manager.data_context_presented:
            return ""

        if not hasattr(self.manager, "file_contents") or not self.manager.file_contents:
            return ""

        sections = []
        sections.append("### Input Files")
        sections.append("The user has provided the following files for context:")
        sections.append("")
        sections.append(self.manager.get_file_contents_text())

        return "\n".join(sections)

    def _generate_tutorial_section(self) -> str:
        """
        Generate a tutorial section if tutorials are available.

        Returns:
            Tutorial section
        """
        if not hasattr(self.manager, "tutorial_prompt"):
            return ""

        tutorial_text = self.manager.tutorial_prompt
        if not tutorial_text or tutorial_text.strip() == "":
            return ""

        sections = []
        sections.append("### Relevant Knowledge and Tutorials")
        sections.append("Here are some relevant tutorials and examples:")
        sections.append("")
        sections.append(tutorial_text)

        return "\n".join(sections)

    def parse(self, response: Dict) -> str:
        """
        Parse the LLM response.

        Args:
            response: The LLM response dictionary

        Returns:
            The parsed response as a string
        """
        if isinstance(response, dict):
            # Handle different response formats
            if "content" in response:
                return response["content"]
            elif "text" in response:
                return response["text"]
            elif "message" in response:
                return response["message"]
            else:
                # Try to get the first value
                for key in ["output", "response", "answer"]:
                    if key in response:
                        return response[key]
                # If nothing matches, convert to string
                return str(response)
        elif isinstance(response, str):
            return response
        else:
            return str(response)
