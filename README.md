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

## Basic Usage

### CLI UI

![Demo](https://github.com/autogluon/autogluon-assistant/blob/main/docs/assets/cli_demo.gif)

```bash
mlzero -i <input_data_folder> [-u <optional_user_instructions>]
```

## WEB UI

![Demo](https://github.com/autogluon/autogluon-assistant/blob/main/docs/assets/web_demo.gif)

```bash
mlzero-backend # command to start backend
mlzero-frontend # command to start frontend on 8509(default)
```

1. **Configure**: Set your model provider and credentials in settings
2. **Upload & Describe**: Drag your data folder into the chat input box, then type what you want to accomplish and press Enter

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
