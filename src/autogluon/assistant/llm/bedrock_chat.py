import logging
from typing import Any, Dict, List

import boto3
from langchain_aws import ChatBedrock

from .base_chat import BaseAssistantChat

logger = logging.getLogger(__name__)


class AssistantChatBedrock(ChatBedrock, BaseAssistantChat):
    """Bedrock chat model with LangGraph support."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.initialize_conversation(self)

    def describe(self) -> Dict[str, Any]:
        base_desc = super().describe()
        return {**base_desc, "model": self.model_id}


def get_bedrock_models() -> List[str]:
    try:
        bedrock = boto3.client("bedrock", region_name="us-west-2")
        response = bedrock.list_foundation_models()
        return [model["modelId"] for model in response["modelSummaries"]]
    except Exception as e:
        logger.error(f"Error fetching Bedrock models: {e}")
        return []


def create_bedrock_chat(config, session_name: str) -> AssistantChatBedrock:
    """Create a Bedrock chat model instance."""
    model = config.model

    logger.info(f"Using Bedrock model: {model} for session: {session_name}")
    return AssistantChatBedrock(
        model_id=model,
        model_kwargs={
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        },
        region_name="us-west-2",
        verbose=config.verbose,
        session_name=session_name,
    )
