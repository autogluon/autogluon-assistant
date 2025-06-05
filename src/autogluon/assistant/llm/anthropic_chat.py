import logging
import os
from typing import Any, Dict, List

from anthropic import Anthropic
from langchain_anthropic import ChatAnthropic

from .base_chat import BaseAssistantChat

logger = logging.getLogger(__name__)


class AssistantChatAnthropic(ChatAnthropic, BaseAssistantChat):
    """Anthropic chat model with LangGraph support."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.initialize_conversation(self)

    def describe(self) -> Dict[str, Any]:
        base_desc = super().describe()
        return {**base_desc, "model": self.model}


def get_anthropic_models() -> List[str]:
    """Get available Anthropic models."""
    try:
        client = Anthropic()
        # List available models
        models = client.models.list()
        print(models)
        return [model.id for model in models.data]
    except Exception as e:
        logger.warning(f"Failed to fetch Anthropic models: {e}")
        # Fallback to hardcoded list
        return []


def create_anthropic_chat(config, session_name: str) -> AssistantChatAnthropic:
    """Create an Anthropic chat model instance."""
    model = config.model

    if "ANTHROPIC_API_KEY" not in os.environ:
        raise ValueError("Anthropic API key not found in environment")

    logger.info(f"Using Anthropic model: {model} for session: {session_name}")
    kwargs = {
        "model": model,
        "anthropic_api_key": os.environ["ANTHROPIC_API_KEY"],
        "session_name": session_name,
        "max_tokens": config.max_tokens,
    }

    if hasattr(config, "temperature"):
        kwargs["temperature"] = config.temperature

    if hasattr(config, "verbose"):
        kwargs["verbose"] = config.verbose

    # Support for additional Anthropic-specific features
    if hasattr(config, "thinking") and hasattr(config.thinking, "enabled"):
        kwargs["thinking"] = config.thinking

    return AssistantChatAnthropic(**kwargs)
