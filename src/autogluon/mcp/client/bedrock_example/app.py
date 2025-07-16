# app.py - Optimized output format version
import asyncio
import json
from datetime import datetime

from autogluon.mcp.client.bedrock_example.converse_agent import ConverseAgent
from autogluon.mcp.client.bedrock_example.converse_tools import ConverseToolManager
from autogluon.mcp.client.bedrock_example.mcp_client import MCPClient
from autogluon.mcp.constants import MCP_BEDROCK_MODEL_ID

# Configuration
PIPELINE_SERVER_URL = "https://your_server_url/mcp/"

# Use a dictionary to avoid scope issues
config = {"debug_mode": True}


def format_response(response_data):
    """Format response data and extract key information"""
    try:
        # Extract basic information
        output = response_data.get("output", {})
        message = output.get("message", {})
        content = message.get("content", [])

        # Extract usage statistics
        usage = response_data.get("usage", {})
        input_tokens = usage.get("inputTokens", 0)
        output_tokens = usage.get("outputTokens", 0)
        total_tokens = usage.get("totalTokens", 0)

        # Extract performance metrics
        latency = response_data.get("metrics", {}).get("latencyMs", 0)

        # Build formatted output
        print("\n" + "=" * 60)
        print("🤖 Assistant Response")
        print("=" * 60)

        # Print message content
        for item in content:
            if "text" in item:
                print(f"\n{item['text']}")
            elif "toolUse" in item:
                tool_info = item["toolUse"]
                print(f"\n🔧 Tool Call: {tool_info.get('name', 'Unknown')}")
                if config["debug_mode"]:
                    print(f"   Input: {json.dumps(tool_info.get('input', {}), indent=2)}")

        # Print statistics information
        print("\n" + "-" * 60)
        print(f"📊 Usage: {input_tokens} in → {output_tokens} out = {total_tokens} total tokens")
        print(f"⏱️  Latency: {latency}ms")
        print("-" * 60)

    except Exception as e:
        print(f"\n❌ Error formatting response: {e}")
        if config["debug_mode"]:
            print("Raw response:", json.dumps(response_data, indent=2))


async def main():
    """
    Main function that sets up and runs an interactive AI agent with tool integration.
    The agent can process user prompts and utilize registered tools to perform tasks.
    """

    # Set up the agent and tool manager
    agent = ConverseAgent(MCP_BEDROCK_MODEL_ID)
    agent.tools = ConverseToolManager()

    # Define the agent's behavior through system prompt
    agent.system_prompt = """You are a helpful assistant that, besides performing any other tasks, is also capable of running AutoGluon ML pipelines. You have access to the run_autogluon_pipeline tool, which can process data and train models. 
When users provide a prompt, first determine whether they want you to use run_autogluon_pipeline. If they do not, respond normally without using this tool. If they do want you to use run_autogluon_pipeline, analyze what they need and ask for any missing required parameters. Once you have received all required parameters, proceed to run run_autogluon_pipeline.
If you run into any errors, please explain in detail at which step the error occurred, what actions you took, what code you executed, and share the exact error message verbatim.
"""

    # Modify ConverseAgent to support formatted output
    original_handle_response = agent._handle_response

    async def formatted_handle_response(response):
        # Display formatted response
        format_response(response)

        # Call the original handler
        return await original_handle_response(response)

    # Replace the method
    agent._handle_response = formatted_handle_response

    async def controlled_invoke(content):
        print("\n👤 User: ", end="")
        if isinstance(content, list) and len(content) > 0:
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    print(item["text"])
                elif isinstance(item, dict) and "toolResult" in item:
                    print(f"[Tool Result from {item['toolResult'].get('toolUseId', 'unknown')}]")
        else:
            print(content)

        # Do not print raw JSON anymore
        agent.messages.append({"role": "user", "content": content})

        response = agent._get_converse_response()

        # Print raw response only in debug mode
        if config["debug_mode"]:
            print(f"\n🔍 Debug - Raw Response: {json.dumps(response, indent=2)}")

        return await agent._handle_response(response)

    # Replace the method
    agent.invoke = controlled_invoke

    # Initialize MCP client with HTTP connection to pipeline server
    try:
        async with MCPClient(PIPELINE_SERVER_URL) as mcp_client:
            # Fetch available tools from the MCP client
            tools = await mcp_client.get_available_tools()

            # Register each available tool with the agent
            for tool in tools:
                agent.tools.register_tool(
                    name=tool["name"],
                    func=mcp_client.call_tool,
                    description=tool["description"],
                    input_schema={"json": tool["inputSchema"]},
                )

            # Clear startup information
            print("\n" + "🚀 " + "=" * 56 + " 🚀")
            print("  AutoGluon MCP Assistant with Bedrock")
            print("  " + "-" * 56)
            print(f"  📡 Connected to: {PIPELINE_SERVER_URL}")
            print(f"  🛠️  Available tools: {', '.join([t['name'] for t in tools])}")
            print(f"  🧠 Model: {MCP_BEDROCK_MODEL_ID.split('.')[-1]}")
            print("  " + "-" * 56)
            print("  Type 'quit' to exit | Toggle debug with 'debug on/off'")
            print("🚀 " + "=" * 56 + " 🚀\n")

            # Start interactive prompt loop
            while True:
                try:
                    # Get user input with timestamp
                    current_time = datetime.now().strftime("%H:%M:%S")
                    user_prompt = input(f"\n[{current_time}] Enter your prompt: ").strip()

                    # Handle commands
                    if user_prompt.lower() in ["quit", "exit", "q"]:
                        print("\n👋 Goodbye!")
                        break
                    elif user_prompt.lower() == "debug on":
                        config["debug_mode"] = True
                        print("🔍 Debug mode enabled")
                        continue
                    elif user_prompt.lower() == "debug off":
                        config["debug_mode"] = False
                        print("🔕 Debug mode disabled")
                        continue
                    elif user_prompt.lower() == "clear":
                        print("\033[2J\033[H")
                        continue

                    # Process the prompt
                    await agent.invoke_with_prompt(user_prompt)

                except KeyboardInterrupt:
                    print("\n\n⚠️  Interrupted! Type 'quit' to exit properly.")
                except Exception as e:
                    print(f"\n❌ Error occurred: {e}")
                    if config["debug_mode"]:
                        import traceback

                        traceback.print_exc()

    except Exception as e:
        print(f"\n❌ Failed to connect to MCP server: {e}")
        print(f"📝 Please ensure the pipeline server is running and accessible at {PIPELINE_SERVER_URL}")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
