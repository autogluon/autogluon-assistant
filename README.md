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

> **Official implementation** of [MLZero: A Multi-Agent System for End-to-end Machine Learning Automation](https://arxiv.org/abs/2505.13941)

AutoGluon Assistant (aka MLZero) is a multi-agent system that automates end-to-end multimodal machine learning or deep learning workflows by transforming raw multimodal data into high-quality ML solutions with zero human intervention. Leveraging specialized perception agents, dual-memory modules, and iterative code generation, it handles diverse data formats while maintaining high success rates across complex ML tasks.

## 💾 Installation

AutoGluon Assistant is supported on Python 3.8 - 3.11 and is available on Linux (will fix dependency issues for MacOS and Windows by our next official release).

You can install from source (new version will be released to PyPI soon):

```bash
pip install uv
uv pip install git+https://github.com/autogluon/autogluon-assistant.git
```

## Quick Start

For detailed usage instructions, OpenAI/Azure setup, and advanced configuration options, see our [Getting Started Tutorial](docs/tutorials/getting_started.md).

## API Setup
MLZero uses AWS Bedrock by default. Configure your AWS credentials:

```bash
export AWS_DEFAULT_REGION="<your-region>"
export AWS_ACCESS_KEY_ID="<your-access-key>"
export AWS_SECRET_ACCESS_KEY="<your-secret-key>"
```

We also support OpenAI. More LLM providers' support (e.g. Anthropic, Azure, etc.) will be added soon.

## Basic Usage for CLI UI

![Demo](https://github.com/autogluon/autogluon-assistant/blob/main/docs/assets/cli_demo.gif)

```bash
mlzero -i <input_data_folder> [-u <optional_user_instructions>]
```

## Basic Usage for WEB UI

![Demo](https://github.com/autogluon/autogluon-assistant/blob/main/docs/assets/web_demo.gif)

### Model Execution Settings

The settings above the divider line control how the model runs, while the settings below the divider line relate to the model being used (including provider, credentials, and model parameters).

### Model Execution Configuration

**Max Iterations**: The number of rounds the model will run. The program automatically stops when this limit is reached. Default is 5, adjustable as needed.

**Manual Prompts Between Iterations**: Choose whether to add iteration-specific prompts between iterations or not.

**Log Verbosity**: Select the level of detail for the logs you want to see. Three options are available: brief, info, and detail. Brief is recommended.
- **Brief**: Contains key essential information
- **Info**: Includes brief information plus detailed information such as file save locations
- **Detail**: Includes info-level information plus all model training related information

### Model Configuration

You can select the LLM provider, model, and credentials to use. If using Bedrock as the provider, you can use EC2 defaults. You can also upload your own config file, which will override the provider and model name settings. Provided credentials will be validated.

### Chat Input Box

1. **Initial Task Submission**: When starting a task for the first time, drag the input folder into this chat input box, enter any description or requirements about the task, then press Enter or click the submit button on the right. Note: Submitting larger files may sometimes fail - you can try multiple times if needed.

2. **Manual Prompts**: If you selected "Manual prompts between iterations" in settings, you can input prompts here.

3. **Task Cancellation**: After submitting a task, if you want to cancel it, submit "cancel" in this input box.


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
