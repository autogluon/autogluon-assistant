import logging
from datetime import datetime
from typing import Dict

from omegaconf import DictConfig

from ..constants import VALID_CODING_LANGUAGES
from ..llm import ChatLLMFactory
from .utils import extract_script

logger = logging.getLogger(__name__)


class LLMCoder:
    """Class to handle code generation using LLM models."""

    def __init__(self, llm_config: DictConfig):
        """Initialize with LLM configuration.

        Args:
            llm_config: Configuration for the LLM model
        """
        self.llm_config = llm_config
        self.multi_turn = llm_config.multi_turn
        if self.multi_turn:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            self.llm = ChatLLMFactory.get_chat_model(llm_config, session_name=f"multi_round_coder_{timestamp}")

    def __call__(self, prompt: str, language: str) -> Dict[str, str]:
        """Generate code using LLM based on prompt.

        Args:
            prompt: The coding prompt
            language: Target programming language

        Returns:
            Dictionary containing full response, language, extracted code
        """
        if not self.multi_turn:
            # create a new session if not multi_turn
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            self.llm = ChatLLMFactory.get_chat_model(self.llm_config, session_name=f"single_round_coder_{timestamp}")

        if language not in VALID_CODING_LANGUAGES:
            raise ValueError(f"Language must be one of {VALID_CODING_LANGUAGES}")

        # Add format instruction if configured
        if hasattr(self.llm_config, "add_coding_format_instruction") and self.llm_config.add_coding_format_instruction:
            format_instruction = f"Please format your response with the code in a ```{language}``` code block to make it easily extractable."
            prompt = f"{prompt}\n\n{format_instruction}"

        # Get response from LLM
        response = self.llm.assistant_chat(prompt)

        # Extract code from response
        code_script = extract_script(response, language.lower())

        return {
            "response": response,
            "language": language.lower(),
            "code_script": code_script,
        }
