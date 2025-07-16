import json
import re

import boto3
from autogluon.mcp.constants import DEFAULT_AWS_REGION


class ConverseAgent:
    def __init__(self, model_id, region=DEFAULT_AWS_REGION, system_prompt="You are a helpful assistant."):
        self.model_id = model_id
        self.region = region
        self.client = boto3.client("bedrock-runtime", region_name=self.region)
        self.system_prompt = system_prompt
        self.messages = []
        self.tools = None
        self.response_output_tags = []  # ['<response>', '</response>']

    async def invoke_with_prompt(self, prompt):
        content = [{"text": prompt}]
        return await self.invoke(content)

    async def invoke(self, content):

        print(f"User: {json.dumps(content, indent=2)}")

        self.messages.append({"role": "user", "content": content})
        response = self._get_converse_response()

        print(f"Agent: {json.dumps(response, indent=2)}")

        return await self._handle_response(response)

    def _get_converse_response(self):
        """
        https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime/client/converse.html
        """

        # print(f"Invoking with messages: {json.dumps(self.messages, indent=2)}")

        response = self.client.converse(
            modelId=self.model_id,
            messages=self.messages,
            system=[{"text": self.system_prompt}],
            inferenceConfig={
                "maxTokens": 8192,
                "temperature": 0.7,
            },
            toolConfig=self.tools.get_tools(),
        )
        return response

    async def _handle_response(self, response):
        # Add the response to the conversation history
        self.messages.append(response["output"]["message"])

        # Do we need to do anything else?
        stop_reason = response["stopReason"]

        if stop_reason in ["end_turn", "stop_sequence"]:
            # Safely extract the text from the nested response structure
            try:
                message = response.get("output", {}).get("message", {})
                content = message.get("content", [])
                text = content[0].get("text", "")
                if hasattr(self, "response_output_tags") and len(self.response_output_tags) == 2:
                    pattern = f"(?s).*{re.escape(self.response_output_tags[0])}(.*?){re.escape(self.response_output_tags[1])}"
                    match = re.search(pattern, text)
                    if match:
                        return match.group(1)
                return text
            except (KeyError, IndexError):
                return ""

        elif stop_reason == "tool_use":
            try:
                # Extract tool use details from response
                tool_response = []
                for content_item in response["output"]["message"]["content"]:
                    if "toolUse" in content_item:
                        tool_request = {
                            "toolUseId": content_item["toolUse"]["toolUseId"],
                            "name": content_item["toolUse"]["name"],
                            "input": content_item["toolUse"]["input"],
                        }

                        tool_result = await self.tools.execute_tool(tool_request)
                        tool_response.append({"toolResult": tool_result})

                return await self.invoke(tool_response)

            except KeyError as e:
                raise ValueError(f"Missing required tool use field: {e}")
            except Exception as e:
                raise ValueError(f"Failed to execute tool: {e}")

        elif stop_reason == "max_tokens":
            # Hit token limit (this is one way to handle it.)
            await self.invoke_with_prompt("Please continue.")

        else:
            raise ValueError(f"Unknown stop reason: {stop_reason}")
