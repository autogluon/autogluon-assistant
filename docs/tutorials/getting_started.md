# MLZero: Getting Started

This guide covers everything you need to know to start using AutoGluon Assistant (MLZero) effectively.

## API Keys

### Configure LLM Providers
MLZero supports AWS Bedrock, Anthropic, OpenAI, and Azure as LLM model providers. You must configure API keys for your chosen provider. MLZero uses AWS Bedrock as the default provider for language models.

You can modify the provider and LLM model in [our default config](https://github.com/FANGAreNotGnu/autogluon-assistant/blob/main/src/autogluon/assistant/configs/default.yaml), or provide a custom config to override the defaults. Different LLM models can be assigned to individual agents by overriding the `default_llm` configuration.

```yaml
llm: &default_llm
  provider: bedrock  # bedrock/anthropic/openai/azure
  model: "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
```

Alternatively, Web UI users can configure providers via the settings panel on the left sidebar.

#### AWS Bedrock Setup
MLZero integrates with AWS Bedrock by default. To use AWS Bedrock, you will need to configure your AWS credentials and region settings:

```bash
export AWS_DEFAULT_REGION="<your-region>"
export AWS_ACCESS_KEY_ID="<your-access-key>"
export AWS_SECRET_ACCESS_KEY="<your-secret-key>"
```

Ensure you have an active AWS account and appropriate permissions set up for using Bedrock models. You can manage your AWS credentials through the AWS Management Console. See [Bedrock supported AWS regions](https://docs.aws.amazon.com/bedrock/latest/userguide/models-regions.html)

#### Anthropic Setup
To use Anthropic, you will need to set your Anthropic API key as an environment variable:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```
You can create an Anthropic account [here](https://console.anthropic.com/) and manage your API keys in the [Console](https://console.anthropic.com/keys).

#### OpenAI Setup
To use OpenAI, you will need to set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY="sk-..."
```

You can sign up for an OpenAI account [here](https://platform.openai.com/) and manage your API keys [here](https://platform.openai.com/account/api-keys).

Important: Free-tier OpenAI accounts may be subject to rate limits, which could affect the performance. We recommend using a paid OpenAI API key for seamless functionality.

#### Azure OpenAI Setup
To use Azure OpenAI, you'll need to set the following Azure OpenAI values, as environment variables:
```bash
export AZURE_OPENAI_API_KEY=<...>
export OPENAI_API_VERSION=<...>
export AZURE_OPENAI_ENDPOINT=<...>
```

## Usage

We support two ways of using AutoGluon Assistant: CLI or WebUI.

### CLI

```bash
mlzero -i INPUT_DATA_FOLDER [-o OUTPUT_DIR] [-c CONFIG_PATH] [-n MAX_ITERATIONS] [--ENABLE-PER-ITERATION-INSTRUCTION] [-t --INITIAL-INSTRUCTION] [-e EXTRACT_TO] [-v VERBOSITY_LEVEL]
```

#### Required Arguments

- `-i, --input`:  
  Path to the input data folder. This directory should contain training/testing files and optionally a description file.

#### Optional Arguments

- `-o, --output`:  
  Path to the output directory. If not specified, a timestamped folder under `runs/` will be automatically generated.

- `-c, --config`:  
  Path to the custom YAML configuration file. If provided, will override default config.

- `-n, --max-iterations`:  
  Maximum number of iterations. Default is `5`.

- `--enable-per-iteration-instruction`:  
  If enabled, provide an instruction at the start of each iteration (except the first, which uses the initial instruction). The process suspends until you provide it.

- `-t, --initial-instruction`:  
  Initial user input to use in the first iteration.

- `-e, --extract-to`:  
  Extract archive files from the input folder to the specified directory and copy all non-archive files to the same directory. If not specified, all data remains in the input folder and is used as-is (archives remain packed).

- `-v, --verbosity`:  
  Increase logging verbosity level. Use `-v <level>` where level is an integer:
  | `-v` value | Logging Level |
  |------------|----------------|
  | 0 | ERROR |
  | 1 (Default) | BRIEF |
  | 2 | INFO |
  | 3 | DETAIL |
  | 4 or more | DEBUG |

#### Examples

```bash
# Minimum usage
mlzero -i <input_data>

# Custom output directory and verbosity
mlzero -i <input_data> -o <output_dir> -v 3

# Copy data and extract archives from input_data to tmp_extract and set max iteration to 3
mlzero -i ./data -e ./tmp_extract -n 3

```

### Web UI

#### Launch Web UI
```bash
mlzero-backend # command to start backend
mlzero-frontend # command to start frontend on 8509(default)
```

#### LLM Configuration
*Note: This configuration is not needed if you exported the required environment variables before starting the Web UI.*

You can select the LLM provider, model, and credentials to use. If using Bedrock as the provider and an EC2 instance as the server, you can also apply the correct IAM role to the EC2 instance.

If you upload your own config file in advanced settings, it will override the provider and model name settings. Provided credentials will be validated.

#### Chat
1. **Upload Data**: When starting a task for the first time, drag the input folder into the chat input box, (optionally) enter any description or requirements about the task, then press Enter or click the submit button on the right. Note: Submitting very large files may sometimes fail due to connection issues - you can try multiple times.
2. **Per Iteration Instruction**: If you selected "Manual prompts between iterations" in the advanced settings, you can input instructions here between iterations.
3. **Cancel The Task**: After submitting a task, if you want to cancel it, submit "cancel" in this chat box.

#### Advanced Settings (Optional)
- **Max Iterations**: MLZero stops when the task is successful or this limit is reached. Default is 5, adjustable as needed.
- **Manual Prompts Between Iterations**: Choose whether to add iteration-specific prompts between iterations.
- **Log Verbosity**: Select the level of detail for the logs you want to see. Three options are available: brief, info, and detail.

### MCP (Model Context Protocol)

#### Local Setup (All services on the same machine)

1. `mlzero-backend`
2. `mlzero-mcp-server` # default port 8000, you can specify a different port using --server-port or -s
3. `mlzero-mcp-client` # default port 8005, you can specify a different port using --port or -p
4. Add MCP server to LLM (e.g. `claude mcp add --transport http <your-server-name> http://localhost:<your-port-number>/mcp/`)
5. Start and watch logs in mlzero-mcp-client terminal (tell LLM the input and output folders)

#### Remote Setup (MCP tools on remote machine, calling from local)

##### On the machine running MCP tools (e.g. EC2)

1. `mlzero-backend`
2. `mlzero-mcp-server`
3. Tunnel for mlzero-mcp-server (e.g. `ngrok http 8000`)

##### On the machine calling MCP (e.g. Mac)

1. Set up SSH, ensure you can `ssh <username>@<ip-or-dns>` (e.g. ssh ubuntu@your-ec2-ip) without providing pem
2. `mlzero-mcp-client --server <username>@<ip-or-dns>` # e.g.: mlzero-mcp-client --server ubuntu@your-ec2-ip
3. Add MCP server to LLM (e.g. `claude mcp add --transport http <your-server-name> http://localhost:<your-port-number>/mcp/`)
4. Start and watch logs in mlzero-mcp-client terminal (in this mode, besides input and output folders, also tell LLM the tunneled address of mlzero-mcp-server)
