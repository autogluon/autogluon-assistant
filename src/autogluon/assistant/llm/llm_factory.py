import logging
import os
from typing import Any, Dict, Optional, Union

from omegaconf import DictConfig

from .anthropic_chat import AssistantChatAnthropic, create_anthropic_chat, get_anthropic_models
from .azure_openai_chat import AssistantAzureChatOpenAI, create_azure_openai_chat, get_azure_models
from .base_chat import GlobalTokenTracker
from .bedrock_chat import AssistantChatBedrock, create_bedrock_chat, get_bedrock_models
from .openai_chat import AssistantChatOpenAI, create_openai_chat, get_openai_models

logger = logging.getLogger(__name__)


class ChatLLMFactory:
    """Factory class for creating chat models with LangGraph support."""

    @staticmethod
    def get_total_token_usage(save_path: Optional[str] = None) -> Dict[str, Any]:
        """Get total token usage across all conversations and sessions."""
        return GlobalTokenTracker().get_total_usage(save_path)

    @classmethod
    def get_valid_models(cls, provider):
        if provider == "azure":
            return get_azure_models()
        elif provider == "openai":
            return get_openai_models()
        elif provider == "bedrock":
            return get_bedrock_models()
        elif provider == "anthropic":
            return get_anthropic_models()
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    @classmethod
    def get_valid_providers(cls):
        return ["azure", "openai", "bedrock", "anthropic"]

    @classmethod
    def get_chat_model(
        cls, config: DictConfig, session_name: str
    ) -> Union[AssistantChatOpenAI, AssistantAzureChatOpenAI, AssistantChatBedrock, AssistantChatAnthropic]:
        """Get a configured chat model instance using LangGraph patterns."""
        provider = config.provider
        model = config.model

        valid_providers = cls.get_valid_providers()
        if provider not in valid_providers:
            raise ValueError(f"Invalid provider: {provider}. Must be one of {valid_providers}")

        valid_models = cls.get_valid_models(provider)
        if model not in valid_models:
            if model[3:] not in valid_models:  # TODO: better logic for cross region inference
                raise ValueError(
                    f"Invalid model: {model} for provider {provider}. All valid models are {valid_models}. If you are using Bedrock, please check if the requested model is available in the provided AWS_DEFAULT_REGION: {os.environ.get('AWS_DEFAULT_REGION')}"
                )

        if provider == "openai":
            return create_openai_chat(config, session_name)
        elif provider == "azure":
            return create_azure_openai_chat(config, session_name)
        elif provider == "anthropic":
            return create_anthropic_chat(config, session_name)
        elif provider == "bedrock":
            return create_bedrock_chat(config, session_name)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
