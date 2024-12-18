<table>
<tr>
<td width="70%">

# AutoGluon Assistant
[![Python Versions](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue)](https://pypi.org/project/autogluon.assistant/)
[![GitHub license](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)
[![Continuous Integration](https://github.com/autogluon/autogluon-assistant/actions/workflows/continuous_integration.yml/badge.svg)](https://github.com/autogluon/autogluon-assistant/actions/workflows/continuous_integration.yml)

</td>
<td>
<img src="https://user-images.githubusercontent.com/16392542/77208906-224aa500-6aba-11ea-96bd-e81806074030.png" width="350">
</td>
</tr>
</table>

AutoGluon Assistant (AG-A) provides users a simple interface where they can input their data, describe their problem, and receive a highly accurate and competitive ML solution — without writing any code. By leveraging the state-of-the-art AutoML capabilities of [AutoGluon](https://github.com/autogluon/autogluon) and integrating them with a Large Language Model (LLM), AG-A automates the entire data science pipeline. AG-A takes [AutoGluon](https://github.com/autogluon/autogluon)'s automation from three lines of code to zero, enabling users to solve new supervised learning tabular problems using only natural language descriptions.

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


#### Beta Features

AG-A now supports automatic feature generation as part of its beta features. To enable these features, please install the beta version dependencies using the following command:

```bash
pip install -r requirements.txt
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


#### Azure OpenAI Setup
To use Azure OpenAI, you'll need to set the following Azure OpenAI values, as environment variables:
```bash
export AZURE_OPENAI_API_KEY=<...>
export OPENAI_API_VERSION=<...>
export AZURE_OPENAI_ENDPOINT=<...>
```

## Usage

We support two ways of using AutoGluon Assistant: WebUI and CLI.

### Web UI
AutoGluon Assistant Web UI allows users to leverage the capabilities of AG-A through an intuitive web interface.

The web UI enables users to upload datasets, configure AG-A runs with customized settings, preview data, monitor execution progress, view and download results, and supports secure, isolated sessions for concurrent users.

#### To run the AG-A Web UI:

```bash
aga ui

# OR

# Launch Web-UI on specific port e.g. 8888
aga ui --port 8888
```

AG-A Web UI should now be accessible in your web browser at `http://localhost:8501` or the specified port.


### CLI

Before launching AG-A CLI, prepare your data files in the following structure:

```
└── data # Data files directory
    ├── train.[ext] # Training dataset (required)
    ├── test.[ext]  # Test dataset (required)
    └── description.txt # Dataset and task description (recommended)
```

Note:
- The training and test files can be in any tabular data format (e.g., csv, parquet, xlsx)
- While there are no strict naming requirements, we recommend using clear, descriptive filenames
- The description file is optional but recommended for better model selection and optimization. It can include:
  - Dataset description
  - Problem context
  - Evaluation metrics
  - Any other relevant information

Now you can launch the AutoGluon Assistant run using the following command:

```bash
aga run [NAME_OF_DATA_DIR] --presets [PRESET_QUALITY]
# e.g. aga run ./toy_data --presets best_quality
```

We support three presets, including `medium_quality`, `high_quality` and `best_quality`. We use `best_quality` as a default setting.

After the run is complete, model predictions on test dataset are saved into the `aga-output-<timestamp>.csv` file. It will be formatted according to optional `sample_submission.csv` file if provided.

#### Overriding Configs

You can override specific settings in the YAML configuration defined in the [config folder](https://github.com/autogluon/autogluon-assistant/tree/main/src/autogluon/assistant/configs) using
the `config_overrides` parameter with format `"key1=value1, key2.nested=value2"` from the command line.


Here are some example commands on using configuration overrides:

```bash
aga run toy_data --config_overrides "feature_transformers.enabled_models=None, time_limit=3600"

# OR

aga run toy_data --config_overrides "feature_transformers.enabled_models=None" --config_overrides "time_limit=3600"
```
