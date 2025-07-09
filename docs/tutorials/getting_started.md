# MLZero: Getting Started

This guide covers everything you need to know to start using AutoGluon Assistant (MLZero) effectively.

## API Keys

### Configuring LLMs
MLZero supports using both AWS Bedrock and OpenAI as LLM model providers. You will need to set up API keys for the respective provider you choose. By default, MLZero uses AWS Bedrock for its language models.

### AWS Bedrock Setup
MLZero integrates with AWS Bedrock by default. To use AWS Bedrock, you will need to configure your AWS credentials and region settings:

```bash
export AWS_DEFAULT_REGION="<your-region>"
export AWS_ACCESS_KEY_ID="<your-access-key>"
export AWS_SECRET_ACCESS_KEY="<your-secret-key>"
```

Ensure you have an active AWS account and appropriate permissions set up for using Bedrock models. You can manage your AWS credentials through the AWS Management Console. See [Bedrock supported AWS regions](https://docs.aws.amazon.com/bedrock/latest/userguide/models-regions.html)

### OpenAI Setup
To use OpenAI, you will need to set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY="sk-..."
```

You can sign up for an OpenAI account [here](https://platform.openai.com/) and manage your API keys [here](https://platform.openai.com/account/api-keys).

Important: Free-tier OpenAI accounts may be subject to rate limits, which could affect AG-A's performance. We recommend using a paid OpenAI API key for seamless functionality.

### Azure OpenAI Setup (WIP)
To use Azure OpenAI, you'll need to set the following Azure OpenAI values, as environment variables:
```bash
export AZURE_OPENAI_API_KEY=<...>
export OPENAI_API_VERSION=<...>
export AZURE_OPENAI_ENDPOINT=<...>
```

## Usage

We support two ways of using AutoGluon Assistant: WebUI or CLI.

### Web UI

#### Model Execution Settings

The settings above the divider line control how the model runs, while the settings below the divider line relate to the model being used (including provider, credentials, and model parameters).

- **Max Iterations**: The number of rounds the model will run. The program automatically stops when this limit is reached. Default is 5, adjustable as needed.

- **Manual Prompts Between Iterations**: Choose whether to add iteration-specific prompts between iterations or not.

- **Log Verbosity**: Select the level of detail for the logs you want to see. Three options are available: brief, info, and detail. Brief is recommended.
  - **Brief**: Contains key essential information
  - **Info**: Includes brief information plus detailed information such as file save locations
  - **Detail**: Includes info-level information plus all model training related information


You can select the LLM provider, model, and credentials to use. If using Bedrock as the provider, you can use EC2 defaults(If you applied the correct IAM role). You can also upload your own config file, which will override the provider and model name settings. Provided credentials will be validated.

#### Chat Input Box

1. **Initial Task Submission**: When starting a task for the first time, drag the input folder into this chat input box, enter any description or requirements about the task, then press Enter or click the submit button on the right. Note: Submitting larger files may sometimes fail - you can try multiple times if needed.

2. **Manual Prompts**: If you selected "Manual prompts between iterations" in settings, you can input instructions here between iterations.

3. **Task Cancellation**: After submitting a task, if you want to cancel it, submit "cancel" in this input box.

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
  Initial user input to use in the first iteration. Optional.

- `-e, --extract-to`:  
  If the input folder contains archive files, unpack them into this directory. If not specified, archives are not unpacked.

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
# Basic usage
mlzero -i ./data

# Custom output directory and verbosity
mlzero -i ./data -o ./results -v 5

# Use archive extraction and limit iterations
mlzero -i ./data -n 3 -e ./tmp_extract -v 6

# Custom output directory and set limit to 3 iterations
mlzero -i ./data -n 3
```
