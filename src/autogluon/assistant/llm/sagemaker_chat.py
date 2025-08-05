import json
import logging
import os
import subprocess
from typing import Any, Dict, List, Optional

import boto3
from botocore.config import Config
from langchain.llms.sagemaker_endpoint import SagemakerEndpoint
from langchain.chat_models.base import BaseChatModel
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import LLMResult

from .base_chat import BaseAssistantChat

logger = logging.getLogger(__name__)


class SagemakerEndpointChat(LLM, BaseAssistantChat):
    """SageMaker endpoint chat model with LangGraph support."""
    
    endpoint_name: str
    inference_component_name: Optional[str] = None
    region_name: str = "us-west-2"
    model_kwargs: Dict[str, Any] = {}
    creds_file: Optional[str] = None
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.initialize_conversation(self)
    
    def _llm_type(self) -> str:
        return "sagemaker_endpoint_chat"
    
    def _get_sagemaker_runtime(self):
        # Refresh credentials before getting the client if creds_file is provided
        if self.creds_file:
            refresh_aws_credentials(self.creds_file)
        
        # Create a new session to force credential refresh
        session = boto3.Session()
        
        boto_config = Config(read_timeout=300)
        return session.client(
            "sagemaker-runtime", 
            region_name=self.region_name,
            config=boto_config
        )
    
    def _process_output_content(self, result: Dict[str, Any]) -> str:
        """Process the model output to combine all content into a single string.
        
        Args:
            result: The raw response from SageMaker endpoint
        
        Returns:
            str: Combined text from the response
        """
        combined_text = ""
        
        # Check if result has 'output' field (new format)
        if 'output' in result:
            for item in result['output']:
                if item.get('type') == 'reasoning':
                    # Skip reasoning content in the final output
                    pass
                elif item.get('type') == 'message' and 'content' in item:
                    # Extract text from message content
                    for content_item in item['content']:
                        if content_item.get('type') == 'output_text':
                            combined_text += content_item.get('text', '')
        
        # If no 'output' field, check for 'predictions' (old format)
        elif 'predictions' in result:
            for prediction in result['predictions']:
                combined_text += prediction.get('text', '')
        
        return combined_text.strip()
    
    def _extract_token_usage(self, response: Dict[str, Any]) -> Dict[str, int]:
        """Extract token usage from response if available."""
        input_tokens = response.get('input_token_count', 0)
        output_tokens = response.get('output_token_count', 0)
        
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens
        }
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> str:
        sagemaker_runtime = self._get_sagemaker_runtime()
        
        payload = {
            "model": "/opt/ml/model",
            "input": prompt,
            "max_num_tokens": self.model_kwargs.get("max_tokens", 2048),
            "stream": False,
            "temperature": self.model_kwargs.get("temperature", 0.5),
            "top_p": self.model_kwargs.get("top_p", 0.9)
        }
        
        invoke_kwargs = {
            "EndpointName": self.endpoint_name,
            "ContentType": "application/json",
            "Accept": "application/json",
            "Body": self._serialize_payload(payload)
        }
        
        # Add InferenceComponentName if provided
        if self.inference_component_name:
            invoke_kwargs["InferenceComponentName"] = self.inference_component_name
        
        response = sagemaker_runtime.invoke_endpoint(**invoke_kwargs)
        result = self._deserialize_response(response)
        
        # Extract the text from the response
        return self._process_output_content(result)
    
    def _serialize_payload(self, payload: Dict[str, Any]) -> bytes:
        """Serialize the payload to bytes."""
        import json
        return json.dumps(payload).encode('utf-8')
    
    def _deserialize_response(self, response: Any) -> Dict[str, Any]:
        """Deserialize the response from bytes."""
        import json
        return json.loads(response['Body'].read().decode())
    
    def describe(self) -> Dict[str, Any]:
        base_desc = super().describe()
        return {
            **base_desc,
            "endpoint_name": self.endpoint_name,
            "inference_component_name": self.inference_component_name,
            "region": self.region_name
        }


def refresh_aws_credentials(creds_file=None):
    """Load fresh credentials from file that's updated by external process.
    
    This function will load credentials from the credentials file that's being
    updated by the external refresh process using STS assume-role.
    
    Args:
        creds_file: Path to the credentials file. If None, credentials won't be refreshed.
    """
    if not creds_file:
        logger.info("No credentials file specified, skipping credential refresh")
        return
        
    # Always load fresh credentials from file, don't use environment variables
    # Clear existing credentials to force reload
    for key in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"]:
        if key in os.environ:
            del os.environ[key]
        
    # Look for credentials.txt file that's updated by external process
    try:
        if os.path.exists(creds_file):
            logger.info(f"Loading AWS credentials from {creds_file}")
            with open(creds_file, "r") as f:
                creds_data = json.load(f)
                
            if "Credentials" in creds_data:
                # Set environment variables from the freshly read file
                os.environ["AWS_ACCESS_KEY_ID"] = creds_data["Credentials"]["AccessKeyId"]
                os.environ["AWS_SECRET_ACCESS_KEY"] = creds_data["Credentials"]["SecretAccessKey"]
                os.environ["AWS_SESSION_TOKEN"] = creds_data["Credentials"]["SessionToken"]
                logger.info("Successfully loaded AWS credentials from file")
            else:
                logger.warning(f"Credentials file exists but has invalid format: {creds_file}")
        else:
            logger.error(f"Credentials file not found: {creds_file}")
    except Exception as e:
        logger.error(f"Error loading AWS credentials from file: {e}")

def get_sagemaker_endpoints(creds_file=None) -> List[str]:
    # Refresh credentials before making the API call if creds_file is provided
    if creds_file:
        refresh_aws_credentials(creds_file)
    
    try:
        # Create a new session to force credential refresh
        session = boto3.Session()
        sagemaker_client = session.client("sagemaker")
        response = sagemaker_client.list_endpoints()
        return [endpoint["EndpointName"] for endpoint in response.get("Endpoints", [])]
    except Exception as e:
        logger.error(f"Error fetching SageMaker endpoints: {e}")
        return []


def create_sagemaker_chat(config, session_name: str) -> SagemakerEndpointChat:
    """Create a SageMaker endpoint chat model instance."""
    endpoint_name = config.endpoint_name
    inference_component_name = config.get("inference_component_name", None)
    creds_file = config.get("creds_file", None)
    region_name = config.get("region_name", "us-west-2")
    
    logger.info(f"Using SageMaker endpoint: {endpoint_name} for session: {session_name}")

    return SagemakerEndpointChat(
        endpoint_name=endpoint_name,
        inference_component_name=inference_component_name,
        model_kwargs={
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "top_p": config.get("top_p", 0.9),
        },
        region_name=region_name,
        verbose=config.verbose,
        session_name=session_name,
        creds_file=creds_file,
    )
