<table>
<tr>
<td width="70%">

# AutoGluon Assistant (aka MLZero)
[![Python Versions](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue)](https://pypi.org/project/autogluon.assistant/)
[![GitHub license](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)
[![Continuous Integration](https://github.com/autogluon/autogluon-assistant/actions/workflows/continuous_integration.yml/badge.svg)](https://github.com/autogluon/autogluon-assistant/actions/workflows/continuous_integration.yml)

</td>
<td>
<img src="https://user-images.githubusercontent.com/16392542/77208906-224aa500-6aba-11ea-96bd-e81806074030.png" width="350">
</td>
</tr>
</table>

AutoGluon Assistant (aka MLZero) is a multi-agent system that automates end-to-end multimodal machine learning or deep learning workflows by transforming raw multimodal data into high-quality ML solutions with zero human intervention. Leveraging specialized perception agents, dual-memory modules, and iterative code generation, it handles diverse data formats while maintaining high success rates across complex ML tasks.

<p align="center">
  <img src="https://github.com/user-attachments/assets/0f0f202e-9804-433b-928a-928cee8ff7fd" alt="aga_demo">
</p>


## 💾 Installation

AutoGluon Assistant is supported on Python 3.8 - 3.11 and is available on Linux, MacOS, and Windows.

You can install with:

```bash
pip install autogluon.assistant
```

You can also install from source:

```bash
git clone https://github.com/autogluon/autogluon-assistant.git
cd autogluon-assistant && pip install -e "."
```

### API Keys

#### Configuring LLMs
AG-A supports using both AWS Bedrock and OpenAI as LLM model providers. You will need to set up API keys for the respective provider you choose. By default, AG-A uses AWS Bedrock for its language models.

#### AWS Bedrock Setup
AG-A integrates with AWS Bedrock by default. To use AWS Bedrock, you will need to configure your AWS credentials and region settings:

```bash
export AWS_DEFAULT_REGION="<your-region>"
export AWS_ACCESS_KEY_ID="<your-access-key>"
export AWS_SECRET_ACCESS_KEY="<your-secret-key>"
```

Ensure you have an active AWS account and appropriate permissions set up for using Bedrock models. You can manage your AWS credentials through the AWS Management Console. See [Bedrock supported AWS regions](https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-regions.html)


#### OpenAI Setup
To use OpenAI, you'll need to set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY="sk-..."
```

You can sign up for an OpenAI account [here](https://platform.openai.com/) and manage your API keys [here](https://platform.openai.com/account/api-keys).

Important: Free-tier OpenAI accounts may be subject to rate limits, which could affect AG-A's performance. We recommend using a paid OpenAI API key for seamless functionality.


#### Azure OpenAI Setup (WIP)
To use Azure OpenAI, you'll need to set the following Azure OpenAI values, as environment variables:
```bash
export AZURE_OPENAI_API_KEY=<...>
export OPENAI_API_VERSION=<...>
export AZURE_OPENAI_ENDPOINT=<...>
```

## Usage

We support two ways of using AutoGluon Assistant: WebUI or CLI.

### Web UI
WIP

### CLI (Better CLI WIP)

The main script `run.py` provides a command-line interface with the following options:

```bash
python run.py -i INPUT_DATA_FOLDER -o OUTPUT_DIR -c CONFIG_PATH [-n MAX_ITERATIONS] [--need_user_input]
```

Arguments:
- `-i, --input_data_folder`: Path to the folder containing input data (required)
- `-o, --output_dir`: Path to the output directory for generated files (required)
- `-c, --config_path`: Path to the configuration file (required)
- `-n, --max_iterations`: Maximum number of iterations for code generation (default: 5)
- `--need_user_input`: Enable user input between iterations (optional flag)

Example:
```bash
python run.py -i ./data -o ./output -c config.yaml -n 3
```


#### Overriding Configs
WIP


## Citation
(Will be released soon) MLZero: A Multi-Agent System for End-to-end Machine Learning Automation
