import logging
import os
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
    if "AWS_DEFAULT_REGION" not in os.environ:
        os.environ["AWS_DEFAULT_REGION"] = "us-west-2"
        logger.info("AWS_DEFAULT_REGION not found is os.environ. Set to default value us-west-2.")

    try:
        bedrock = boto3.client("bedrock")
        response = bedrock.list_foundation_models()
        return [model["modelId"] for model in response["modelSummaries"]]
    except Exception as e:
        logger.error(f"Error fetching Bedrock models: {e}")
        return []


def create_bedrock_chat(config, session_name: str) -> AssistantChatBedrock:
    """Create a Bedrock chat model instance."""
    model = config.model

    logger.info(f"Using Bedrock model: {model} for session: {session_name}")
    if "AWS_DEFAULT_REGION" not in os.environ:
        raise ValueError("AWS_DEFAULT_REGION key not found in environment")

    return AssistantChatBedrock(
        model_id=model,
        model_kwargs={
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        },
        region_name=os.environ["AWS_DEFAULT_REGION"],
        verbose=config.verbose,
        session_name=session_name,
    )
