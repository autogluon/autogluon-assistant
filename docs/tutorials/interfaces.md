# Using Different Interfaces

AutoGluon Assistant offers multiple ways to interact with its capabilities. This tutorial covers the available interfaces and how to use them effectively.

## Available Interfaces

AutoGluon Assistant provides four main interfaces:

1. **Command Line Interface (CLI)** - For quick and scriptable interactions
2. **Python API** - For programmatic integration into your code
3. **Web UI** - For a visual, user-friendly experience
4. **MCP (Model Context Protocol)** - For integration with LLM interfaces

## Command Line Interface (CLI)

The CLI offers a quick way to run AutoGluon Assistant from your terminal.

### Basic Usage

```bash
mlzero -i <input_data_folder> [-o <output_dir>] [-t "<initial_instruction>"]
```

### Key CLI Options

| Option | Description |
|--------|-------------|
| **Required** | |
| `-i, --input` | Path to data folder |
| **Optional** | |
| `-o, --output` | Output directory (default: auto-generated under runs/) |
| `-c, --config` | Path to YAML config file |
| `-n, --max-iterations` | Maximum iteration count (default: 5) |
| `--provider` | LLM provider to use (bedrock, openai, anthropic, sagemaker) |
| `-t, --initial-instruction` | Initial user instruction |
| `-v, --verbosity` | Logging verbosity (0-4) |
| `-e, --extract-to` | Copy input data to specified directory and extract all .zip archives |
| `--continuous_improvement` | Continue optimizing after finding a valid solution |
| `--enable-per-iteration-instruction` | Enable user input between iterations
```

### CLI Example

```bash
mlzero -i ./data                        # Path to input data folder (required)
       -o ./my_output                   # Custom output directory
       -t "Train a tabular classifier"   # Initial instruction
       --provider anthropic             # Use Anthropic's Claude models
       -n 3                             # Run maximum 3 iterations
       -v 2                             # More detailed logging
       -e ./extracted_data              # Extract archives and copy data here
       -c ./my_config.yaml              # Use custom configuration
```

## Web UI

The Web UI provides a user-friendly graphical interface for interacting with AutoGluon Assistant.

![Demo](https://github.com/autogluon/autogluon-assistant/blob/main/docs/assets/web_demo.gif)

### Starting the Web UI

```bash
# Start the backend server
mlzero-backend

# In a separate terminal, start the frontend
mlzero-frontend
```

By default, the Web UI will be available at http://localhost:8509.

### Advanced Settings of WebUI
#### Model Execution Settings
The settings above the divider line control how the model runs, while the settings below the divider line relate to the model being used (including provider, credentials, and model parameters).
#### Model Execution Configuration
**Max Iterations**: The number of rounds the model will run. The program automatically stops when this limit is reached. Default is 5, adjustable as needed.
Manual Prompts Between Iterations: Choose whether to add iteration-specific prompts between iterations or not.
**Log Verbosity**: Select the level of detail for the logs you want to see. Three options are available: brief, info, and detail. Brief is recommended.
**Brief**: Contains key essential information
**Info**: Includes brief information plus detailed information such as file save locations
**Detail**: Includes info-level information plus all model training related information
#### Model Configuration
You can select the LLM provider, model, and credentials to use. If using Bedrock as the provider, you can use EC2 defaults. You can also upload your own config file, which will override the provider and model name settings. Provided credentials will be validated.
#### Chat Input Box
**Initial Task Submission**: When starting a task for the first time, drag the input folder into this chat input box, enter any description or requirements about the task, then press Enter or click the submit button on the right. Note: Submitting larger files may sometimes fail - you can try multiple times if needed.
**Manual Prompts**: If you selected "Manual prompts between iterations" in settings, you can input prompts here.
**Task Cancellation**: After submitting a task, if you want to cancel it, submit "cancel" in this input box.

## Python API

The Python API allows you to integrate AutoGluon Assistant directly into your Python code or notebooks.

### Basic Usage

```python
from autogluon.assistant import run_agent

# Simple usage
run_agent(
    input_data_folder="./data",
    output_folder="./output",
    initial_user_input="Create a classification model"
)
```

### Advanced Configuration

```python
from autogluon.assistant import run_agent

# Run with custom settings
run_agent(
    input_data_folder="./data",              # Path to input data folder (required)
    output_folder="./output",               # Custom output directory
    config_path="./my_config.yaml",         # Custom configuration file
    max_iterations=3,                      # Maximum number of iterations
    continuous_improvement=True,           # Continue after success for better solutions
    need_user_input=True,                  # Enable prompts between iterations
    initial_user_input="Create a multi-modal classification model",  # Initial instruction
    extract_archives_to="./extracted_data",  # Extract archives and copy data here
    verbosity=2                            # More detailed logging
)
```

## MCP (Model Context Protocol)

MCP allows you to integrate AutoGluon Assistant with other LLM interfaces like Claude or ChatGPT.

### Local Setup

To run all services on the same machine:

```bash
# 1. Start the backend
mlzero-backend

# 2. Start the MCP server (default port: 8000)
mlzero-mcp-server
# You can specify a different port:
# mlzero-mcp-server --server-port 8001

# 3. Start the MCP client (default port: 8005)
mlzero-mcp-client
# You can specify a different port:
# mlzero-mcp-client --port 8006

# 4. Add MCP server to your LLM interface
# Example for Claude CLI:
claude mcp add --transport http autogluon http://localhost:8000/mcp/

# 5. Now you can interact with AutoGluon through your LLM interface
```

### Remote Setup

For running MCP tools on a remote machine:

#### On the Remote Machine (e.g., EC2)

```bash
# 1. Start backend
mlzero-backend

# 2. Start MCP server
mlzero-mcp-server

# 3. Create a tunnel (e.g., with ngrok)
ngrok http 8000
# Note the generated URL
```

#### On Your Local Machine

```bash
# 1. Ensure SSH access is configured
# ssh <username>@<remote-ip> should work without password

# 2. Start MCP client connected to remote server
mlzero-mcp-client --server <username>@<remote-ip>

# 3. Add MCP server to your LLM interface
# Use the ngrok URL from the remote machine:
claude mcp add --transport http autogluon https://<ngrok-url>/mcp/

# 4. Now you can interact with the remote AutoGluon through your local LLM interface
```

## Choosing the Right Interface

Each interface has specific advantages:

- **CLI**: Best for quick tasks, automation, and scripting
- **Web UI**: Perfect for beginners and those who prefer visual interfaces
- **Python API**: Ideal for integration into existing Python workflows and notebooks
- **MCP**: Great for users already working with LLMs who want to extend capabilities