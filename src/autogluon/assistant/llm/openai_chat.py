import logging
import os
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI
from openai import OpenAI

from .base_chat import BaseAssistantChat

logger = logging.getLogger(__name__)


class AssistantChatOpenAI(ChatOpenAI, BaseAssistantChat):
    """OpenAI chat model with LangGraph support."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.initialize_conversation(self)

    def describe(self) -> Dict[str, Any]:
        base_desc = super().describe()
        return {**base_desc, "model": self.model_name, "proxy": self.openai_proxy}


def get_openai_models() -> List[str]:
    try:
        client = OpenAI()
        models = client.models.list()
        return [model.id for model in models if model.id.startswith(("gpt-3.5", "gpt-4", "o1", "o3"))]
    except Exception as e:
        logger.error(f"Error fetching OpenAI models: {e}")
        return []


def create_openai_chat(config, session_name: str) -> AssistantChatOpenAI:
    """Create an OpenAI chat model instance."""
    model = config.model

    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError("OpenAI API key not found in environment")

    logger.info(f"Using OpenAI model: {model} for session: {session_name}")
    kwargs = {
        "model_name": model,
        "openai_api_key": os.environ["OPENAI_API_KEY"],
        "session_name": session_name,
        "max_tokens": config.max_tokens,
    }

    if hasattr(config, "temperature"):
        kwargs["temperature"] = config.temperature

    if hasattr(config, "verbose"):
        kwargs["verbose"] = config.verbose

    if hasattr(config, "proxy_url"):
        kwargs["openai_api_base"] = config.proxy_url

    return AssistantChatOpenAI(**kwargs)
