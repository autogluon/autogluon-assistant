import logging
import os
from typing import Any, Dict, List, Union

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

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

# Current direct text choices. Explicit saved/custom model IDs remain usable.
MINIMAX_MODELS = ["MiniMax-M3.1-Flash", "MiniMax-M3"]
DEFAULT_MINIMAX_MODEL = "MiniMax-M3.1-Flash"


def _minimax_request_payload(payload: Dict[str, Any], api: str) -> Dict[str, Any]:
    """Normalize current model options after LangChain merges invocation overrides."""
    payload = dict(payload)
    extra = dict(payload.get("extra_body") or {})
    model = extra.get("model", payload.get("model"))
    if model not in MINIMAX_MODELS:
        return payload

    if model == "MiniMax-M3.1-Flash":
        if api == "openai":
            effort = extra.pop("reasoning_effort", payload.get("reasoning_effort", "max"))
            payload["reasoning_effort"] = effort if effort in ("low", "medium", "high", "xhigh", "max") else "max"
        else:
            output_config = dict(payload.pop("output_config", None) or {})
            output_config.update(extra.get("output_config") or {})
            if output_config.get("effort") not in ("low", "medium", "high", "xhigh", "max"):
                output_config["effort"] = "max"
            extra["output_config"] = output_config
        payload.pop("thinking", None)
        extra["thinking"] = {"type": "adaptive"}
        for options in (payload, extra):
            options.pop("service_tier", None)
            options.pop("reasoning_split", None)
    else:
        thinking = extra.get("thinking", payload.get("thinking"))
        if isinstance(thinking, dict):
            kind = thinking.get("type")
            if kind is None and "enabled" in thinking:
                kind = "adaptive" if thinking["enabled"] else "disabled"
            if kind in ("enabled", "adaptive", "disabled"):
                payload.pop("thinking", None)
                extra["thinking"] = {"type": "disabled" if kind == "disabled" else "adaptive"}
    if extra or "extra_body" in payload:
        payload["extra_body"] = extra
    return payload


class AssistantChatMiniMax(ChatOpenAI, BaseAssistantChat):
    """MiniMax chat model over the OpenAI-compatible route with LangGraph support."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.initialize_conversation(self)

    def _get_request_payload(self, input_, *, stop=None, **kwargs) -> Dict[str, Any]:
        return _minimax_request_payload(super()._get_request_payload(input_, stop=stop, **kwargs), "openai")

    def describe(self) -> Dict[str, Any]:
        base_desc = super().describe()
        return {**base_desc, "model": self.model_name, "base_url": self.openai_api_base}


class AssistantChatMiniMaxAnthropic(ChatAnthropic, BaseAssistantChat):
    """MiniMax chat model over the Anthropic-compatible route with LangGraph support."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.initialize_conversation(self)

    def _get_request_payload(self, input_, *, stop=None, **kwargs) -> Dict[str, Any]:
        return _minimax_request_payload(super()._get_request_payload(input_, stop=stop, **kwargs), "anthropic")

    def describe(self) -> Dict[str, Any]:
        base_desc = super().describe()
        return {**base_desc, "model": self.model}


def _resolve_region(region: str) -> Dict[str, str]:
    if region not in MINIMAX_REGIONAL_ENDPOINTS:
        raise ValueError(f"Invalid MiniMax region: {region}. Must be one of {list(MINIMAX_REGIONAL_ENDPOINTS)}")
    return MINIMAX_REGIONAL_ENDPOINTS[region]


def get_minimax_models(region: str = DEFAULT_MINIMAX_REGION) -> List[str]:
    """Return current direct text choices, independent of legacy discovery entries."""
    _resolve_region(region)
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

    request_options = {}
    if getattr(config, "thinking", None) is not None:
        request_options["thinking"] = dict(config.thinking)
    if getattr(config, "reasoning_effort", None) is not None:
        if api == "openai":
            request_options["reasoning_effort"] = config.reasoning_effort
        else:
            request_options["output_config"] = {"effort": config.reasoning_effort}
    if getattr(config, "service_tier", None) is not None:
        request_options["service_tier"] = config.service_tier

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

        if "thinking" in request_options:
            kwargs["thinking"] = request_options.pop("thinking")
        kwargs["model_kwargs"] = request_options
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

    if "thinking" in request_options:
        kwargs["extra_body"] = {"thinking": request_options.pop("thinking")}
    kwargs.update(request_options)
    return AssistantChatMiniMax(**kwargs)
