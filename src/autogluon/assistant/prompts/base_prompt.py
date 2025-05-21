import logging
from abc import ABC, abstractmethod
from typing import Dict

logger = logging.getLogger(__name__)


class BasePrompt(ABC):
    """Abstract base class for prompt handling"""

    def __init__(self, llm_config, template=None):
        """
        Initialize prompt handler with configuration and optional template.

        Args:
            llm_config: Configuration for the language model
            template: Optional custom template. Can be:
                     - None: use default template
                     - A string path ending in .txt: load template from file
                     - A string: use as template directly
        """
        self.llm_config = llm_config
        self.set_template(template)

    def set_template(self, template):
        """
        Set a new template.

        Args:
            template: Can be a file path ending in .txt or a template string
        """
        if template is None:
            self.template = self.default_template()
        elif isinstance(template, str) and template.endswith(".txt"):
            try:
                logger.info(f"Loading template from file {template}")
                with open(template, "r") as f:
                    self.template = f.read()
            except Exception as e:
                logger.warning(f"Failed to load template from file {template}: {e}")
                self.template = self.default_template()
        else:
            self.template = template

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
