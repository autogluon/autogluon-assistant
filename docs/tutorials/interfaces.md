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

```bash
# Required
-i, --input            Path to data folder

# Optional
-o, --output           Output directory (default: auto-generated under runs/)
-c, --config           Path to YAML config file
-n, --max-iterations   Maximum iteration count (default: 5)
--provider             LLM provider to use (bedrock, openai, anthropic, sagemaker)
-t, --initial-instruction  Initial user instruction
-v, --verbosity        Logging verbosity (0-4)
-e, --extract-to       Extract archives to this directory
--continuous_improvement  Continue optimizing after finding a valid solution
--enable-per-iteration-instruction  Enable user input between iterations
```

### CLI Examples

```bash
# Basic usage with an instruction
mlzero -i ./data -t "Create a classification model using AutoGluon"

# Specify output directory and use OpenAI
mlzero -i ./data -o ./my_output --provider openai

# Use custom config and verbose logging
mlzero -i ./data -c ./my_config.yaml -v 3

# Extract archives and limit to 3 iterations
mlzero -i ./data -e ./extracted_data -n 3
```

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
from pathlib import Path
import yaml

# Load custom config
with open("my_config.yaml", "r") as f:
    custom_config = yaml.safe_load(f)

# Run with custom settings
run_agent(
    input_data_folder="./data",
    output_folder="./output",
    config_path="./my_config.yaml",
    max_iterations=3,
    continuous_improvement=True,
    need_user_input=True,
    initial_user_input="Create a classification model",
    verbosity=2
)
```

### Integration Example

```python
import pandas as pd
from autogluon.assistant import run_agent

# Process data
data = pd.read_csv("raw_data.csv")
data.to_csv("processed_data/train.csv", index=False)

# Run AutoGluon Assistant
result = run_agent(
    input_data_folder="./processed_data",
    output_folder="./model_output",
    initial_user_input="Train a classification model and evaluate performance"
)

# Continue your workflow with the generated model
# ...
```

## Web UI

The Web UI provides a user-friendly graphical interface for interacting with AutoGluon Assistant.

### Starting the Web UI

```bash
# Start the backend server
mlzero-backend

# In a separate terminal, start the frontend
mlzero-frontend
```

By default, the Web UI will be available at http://localhost:8509.

### Web UI Features

1. **Chat Interface**: Interact with the assistant conversationally
2. **File Upload**: Drag and drop input data folders
3. **Settings Panel**: Configure LLM providers and parameters
4. **Execution History**: View past runs and their outputs
5. **Real-time Logs**: See detailed execution logs

### Configuration in Web UI

The Web UI allows you to:

1. Select your LLM provider and model
2. Set API credentials
3. Configure advanced settings like max iterations and verbosity
4. Upload custom configuration files

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
- **Python API**: Ideal for integration into existing Python workflows and notebooks
- **Web UI**: Perfect for beginners and those who prefer visual interfaces
- **MCP**: Great for users already working with LLMs who want to extend capabilities

## Best Practices

- **Data Preparation**: Organize your data well regardless of interface
- **Error Handling**: The CLI and Python API provide more detailed error information
- **Resource Monitoring**: For large jobs, the Web UI makes it easier to monitor progress
- **Configuration**: Use YAML files for consistent configuration across interfaces

## Troubleshooting Common Issues

- **Port Conflicts**: If WebUI or MCP ports are already in use, specify alternative ports
- **File Permissions**: Ensure your user has read/write access to input/output directories
- **Environment Variables**: Check that API keys are properly set in your environment
- **Network Issues**: For remote setups, verify firewalls allow necessary connections