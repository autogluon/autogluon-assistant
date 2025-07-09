from typing import Any, Callable, Dict, List


class ConverseToolManager:
    def __init__(self):
        self._tools = {}
        self._name_mapping = {}  # Maps sanitized names to original names

    def _sanitize_name(self, name: str) -> str:
        """Convert hyphenated names to underscore format"""
        return name.replace("-", "_")

    def register_tool(self, name: str, func: Callable, description: str, input_schema: Dict):
        """
        Register a new tool with the system, sanitizing the name for Bedrock compatibility
        """
        sanitized_name = self._sanitize_name(name)
        self._name_mapping[sanitized_name] = name
        self._tools[sanitized_name] = {
            "function": func,
            "description": description,
            "input_schema": input_schema,
            "original_name": name,
        }

    def get_tools(self) -> Dict[str, List[Dict]]:
        """
        Generate the tools specification using sanitized names
        """
        tool_specs = []
        for sanitized_name, tool in self._tools.items():
            tool_specs.append(
                {
                    "toolSpec": {
                        "name": sanitized_name,  # Use sanitized name for Bedrock
                        "description": tool["description"],
                        "inputSchema": tool["input_schema"],
                    }
                }
            )

        return {"tools": tool_specs}

    async def execute_tool(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool based on the agent's request, handling name translation
        """
        tool_use_id = payload["toolUseId"]
        sanitized_name = payload["name"]
        tool_input = payload["input"]

        if sanitized_name not in self._tools:
            raise ValueError(f"Unknown tool: {sanitized_name}")
        try:
            tool_func = self._tools[sanitized_name]["function"]
            # Use original name when calling the actual function
            original_name = self._tools[sanitized_name]["original_name"]
            result = await tool_func(original_name, tool_input)
            return {"toolUseId": tool_use_id, "content": [{"text": str(result)}], "status": "success"}
        except Exception as e:
            return {
                "toolUseId": tool_use_id,
                "content": [{"text": f"Error executing tool: {str(e)}"}],
                "status": "error",
            }

    def clear_tools(self):
        """Clear all registered tools"""
        self._tools.clear()
