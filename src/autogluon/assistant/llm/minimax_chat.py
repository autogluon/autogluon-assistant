import logging
import os
from typing import Any, Dict, List, Union

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from openai import OpenAI

from .base_chat import BaseAssistantChat

logger = logging.getLogger(__name__)

# Regional routes exposed by MiniMax. Each region serves both an
# OpenAI-compatible endpoint and an Anthropic-compatible endpoint.
MINIMAX_REGIONAL_ENDPOINTS = {
    "global_en": {
        "openai_base_url": "https://api.minimax.io/v1",
        "anthropic_base_url": "https://api.minimax.io/anthropic",
    },
    "cn_zh": {
        "openai_base_url": "https://api.minimaxi.com/v1",
        "anthropic_base_url": "https://api.minimaxi.com/anthropic",
    },
}

DEFAULT_MINIMAX_REGION = "global_en"

# Supported compatibility routes.
MINIMAX_APIS = ("openai", "anthropic")
DEFAULT_MINIMAX_API = "openai"

# Current MiniMax chat models, used as a fallback when live discovery is unavailable.
MINIMAX_MODELS = ["MiniMax-M3", "MiniMax-M2.7"]
DEFAULT_MINIMAX_MODEL = "MiniMax-M3"


class AssistantChatMiniMax(ChatOpenAI, BaseAssistantChat):
    """MiniMax chat model over the OpenAI-compatible route with LangGraph support."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.initialize_conversation(self)

    def describe(self) -> Dict[str, Any]:
        base_desc = super().describe()
        return {**base_desc, "model": self.model_name, "base_url": self.openai_api_base}


class AssistantChatMiniMaxAnthropic(ChatAnthropic, BaseAssistantChat):
    """MiniMax chat model over the Anthropic-compatible route with LangGraph support."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.initialize_conversation(self)

    def describe(self) -> Dict[str, Any]:
        base_desc = super().describe()
        return {**base_desc, "model": self.model}


def _resolve_region(region: str) -> Dict[str, str]:
    if region not in MINIMAX_REGIONAL_ENDPOINTS:
        raise ValueError(f"Invalid MiniMax region: {region}. Must be one of {list(MINIMAX_REGIONAL_ENDPOINTS)}")
    return MINIMAX_REGIONAL_ENDPOINTS[region]


def get_minimax_models(region: str = DEFAULT_MINIMAX_REGION) -> List[str]:
    """Get available MiniMax models via the OpenAI-compatible route with a static fallback."""
    endpoints = _resolve_region(region)
    try:
        client = OpenAI(api_key=os.environ.get("MINIMAX_API_KEY"), base_url=endpoints["openai_base_url"])
        models = client.models.list()
        discovered = [model.id for model in models.data if model.id.startswith("MiniMax")]
        if discovered:
            return discovered
    except Exception as e:
        logger.warning(f"Failed to fetch MiniMax models: {e}")
    return list(MINIMAX_MODELS)


def create_minimax_chat(config, session_name: str) -> Union[AssistantChatMiniMax, AssistantChatMiniMaxAnthropic]:
    """Create a MiniMax chat model instance for the selected regional and compatibility route."""
    model = config.model

    if "MINIMAX_API_KEY" not in os.environ:
        raise ValueError("MiniMax API key not found in environment")

    region = getattr(config, "region", DEFAULT_MINIMAX_REGION)
    endpoints = _resolve_region(region)

    api = getattr(config, "api", DEFAULT_MINIMAX_API)
    if api not in MINIMAX_APIS:
        raise ValueError(f"Invalid MiniMax api: {api}. Must be one of {list(MINIMAX_APIS)}")

    api_key = os.environ["MINIMAX_API_KEY"]
    logger.info(f"Using MiniMax model: {model} ({api} route, region {region}) for session: {session_name}")

    if api == "anthropic":
        kwargs = {
            "model": model,
            "anthropic_api_key": api_key,
            "anthropic_api_url": endpoints["anthropic_base_url"],
            "session_name": session_name,
            "max_tokens": config.max_tokens,
        }

        if hasattr(config, "temperature"):
            kwargs["temperature"] = config.temperature

        if hasattr(config, "verbose"):
            kwargs["verbose"] = config.verbose

        if hasattr(config, "thinking") and hasattr(config.thinking, "enabled"):
            kwargs["thinking"] = config.thinking

        return AssistantChatMiniMaxAnthropic(**kwargs)

    kwargs = {
        "model_name": model,
        "openai_api_key": api_key,
        "openai_api_base": endpoints["openai_base_url"],
        "session_name": session_name,
        "max_tokens": config.max_tokens,
    }

    if hasattr(config, "temperature"):
        kwargs["temperature"] = config.temperature

    if hasattr(config, "verbose"):
        kwargs["verbose"] = config.verbose

    return AssistantChatMiniMax(**kwargs)
