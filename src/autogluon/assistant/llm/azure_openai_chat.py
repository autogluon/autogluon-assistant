import logging
import os
from typing import Any, Dict, List

from langchain_openai import AzureChatOpenAI
from openai import AzureOpenAI

from .base_chat import BaseAssistantChat

logger = logging.getLogger(__name__)


class AssistantAzureChatOpenAI(AzureChatOpenAI, BaseAssistantChat):
    """Azure OpenAI chat model with LangGraph support."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.initialize_conversation(self)

    def describe(self) -> Dict[str, Any]:
        base_desc = super().describe()
        return {**base_desc, "model": self.model_name}


def get_azure_models() -> List[str]:
    try:
        client = AzureOpenAI()
        models = client.models.list()
        return [model.id for model in models if model.id.startswith(("gpt-3.5", "gpt-4", "o1", "o3"))]
    except Exception as e:
        print(f"Error fetching Azure models: {e}")
        return []


def create_azure_openai_chat(config, session_name: str) -> AssistantAzureChatOpenAI:
    """Create an Azure OpenAI chat model instance."""
    model = config.model

    logger.info(f"Using Azure OpenAI model: {model} for session: {session_name}")
    if "AZURE_OPENAI_API_KEY" not in os.environ:
        raise ValueError("Azure OpenAI API key not found in environment")
    if "OPENAI_API_VERSION" not in os.environ:
        raise Exception("Azure API env variable OPENAI_API_VERSION not set")
    if "AZURE_OPENAI_ENDPOINT" not in os.environ:
        raise Exception("Azure API env variable AZURE_OPENAI_ENDPOINT not set")

    kwargs = {
        "model_name": model,
        "openai_api_key": os.environ["AZURE_OPENAI_API_KEY"],
        "api_version": os.environ["OPENAI_API_VERSION"],
        "azure_endpoint": os.environ["AZURE_OPENAI_ENDPOINT"],
        "session_name": session_name,
        "max_tokens": config.max_tokens,
    }

    if hasattr(config, "temperature"):
        kwargs["temperature"] = config.temperature

    if hasattr(config, "verbose"):
        kwargs["verbose"] = config.verbose

    return AssistantAzureChatOpenAI(**kwargs)
