# mcp_client.py - Modified for HTTP connection
import json
from typing import Any, List

from fastmcp import Client


class MCPClient:
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.client = None
        self._tools_cache = None

    async def __aenter__(self):
        """Async context manager entry point"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.client:
            await self.client.__aexit__(exc_type, exc_val, exc_tb)

    async def connect(self):
        """Establish connection to the MCP server"""
        try:
            self.client = Client(self.server_url)
            await self.client.__aenter__()
        except Exception as e:
            raise RuntimeError(f"Failed to connect to MCP server at {self.server_url}: {e}")

    async def get_available_tools(self) -> List[Any]:
        """Dynamically retrieve the list of available tools from the server"""
        if not self.client:
            raise RuntimeError("Not connected to MCP server")

        # Return cached tools if already fetched
        if self._tools_cache is not None:
            return self._tools_cache

        # Request the list of tools from the server
        tools = await self.client.list_tools()

        # Convert to the expected format
        formatted_tools = []
        for tool in tools:
            formatted_tools.append(
                {"name": tool.name, "description": tool.description, "inputSchema": tool.inputSchema}
            )

        # Optionally cache the formatted result
        self._tools_cache = formatted_tools

        return formatted_tools

    async def call_tool(self, tool_name: str, arguments: dict) -> Any:
        """Call a tool with the specified arguments"""
        if not self.client:
            raise RuntimeError("Not connected to MCP server")

        result = await self.client.call_tool(tool_name, arguments)

        # Parse the response according to fastmcp format
        if isinstance(result, list) and len(result) > 0:
            return json.loads(result[0].text)
        return result
