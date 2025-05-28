<table>
<tr>
<td width="70%">

# AutoGluon Assistant (aka MLZero)
[![Python Versions](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue)](https://pypi.org/project/autogluon.assistant/)
[![GitHub license](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)
[![Continuous Integration](https://github.com/autogluon/autogluon-assistant/actions/workflows/continuous_integration.yml/badge.svg)](https://github.com/autogluon/autogluon-assistant/actions/workflows/continuous_integration.yml)
[![Project Page](https://img.shields.io/badge/Project_Page-MLZero-blue)](https://project-mlzero.github.io/)

</td>
<td>
<img src="https://user-images.githubusercontent.com/16392542/77208906-224aa500-6aba-11ea-96bd-e81806074030.png" width="350">
</td>
</tr>
</table>

AutoGluon Assistant (aka MLZero) is a multi-agent system that automates end-to-end multimodal machine learning or deep learning workflows by transforming raw multimodal data into high-quality ML solutions with zero human intervention. Leveraging specialized perception agents, dual-memory modules, and iterative code generation, it handles diverse data formats while maintaining high success rates across complex ML tasks.

## 💾 Installation

AutoGluon Assistant is supported on Python 3.8 - 3.11 and is available on Linux, MacOS, and Windows.


You can install from source:

```bash
git clone https://github.com/autogluon/autogluon-assistant.git
cd autogluon-assistant && pip install -e "."
```

You can also install old version with (not recommended):

```bash
pip install autogluon.assistant
```

### API Keys

#### Configuring LLMs
MLZero supports using both AWS Bedrock and OpenAI as LLM model providers. You will need to set up API keys for the respective provider you choose. By default, MLZero uses AWS Bedrock for its language models.

#### AWS Bedrock Setup
MLZero integrates with AWS Bedrock by default. To use AWS Bedrock, you will need to configure your AWS credentials and region settings:

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

### CLI


```bash
mlzero -i INPUT_DATA_FOLDER [-c CONFIG_PATH] [-n MAX_ITERATIONS] [-u INITIAL_USER_INPUT]
```


### Required Arguments

- `-i, --input`:  
  Path to the input data folder. This directory should contain training/testing files and optionally a description file.

### Optional Arguments

- `-c, --config`:  
  Path to the YAML configuration file. Default: `configs/default.yaml`.

- `-n, --max-iterations`:  
  Maximum number of iterations. Default is `5`.

- `-u, --user-input`:  
  Initial user input to use in the first iteration. Optional.

> **Note:** For more optional arguments and examples, see the [CLI tutorial](https://github.com/autogluon/autogluon-assistant/tree/main/docs/tutorials/cli_tutorial.md).

### Examples

```bash
# Basic usage
mlzero -i ./data

# Custom output directory and set limit to 3 iterations
mlzero -i ./data -n 3


#### Overriding Configs
You can always provide a config to override default config.
```


## Citation
If you use Autogluon Assistant (MLZero) in your research, please cite our paper:

```bibtex
@misc{fang2025mlzeromultiagentendtoendmachine,
      title={MLZero: A Multi-Agent System for End-to-end Machine Learning Automation}, 
      author={Haoyang Fang and Boran Han and Nick Erickson and Xiyuan Zhang and Su Zhou and Anirudh Dagar and Jiani Zhang and Ali Caner Turkmen and Cuixiong Hu and Huzefa Rangwala and Ying Nian Wu and Bernie Wang and George Karypis},
      year={2025},
      eprint={2505.13941},
      archivePrefix={arXiv},
      primaryClass={cs.MA},
      url={https://arxiv.org/abs/2505.13941}, 
}
```
