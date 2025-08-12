<table>
<tr>
<td width="70%">

# AutoGluon Assistant (aka MLZero)
[![Python Versions](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue)](https://pypi.org/project/autogluon.assistant/)
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

AutoGluon Assistant (aka MLZero) is a multi-agent system that automates end-to-end multimodal machine learning or deep learning workflows by transforming raw multimodal data into high-quality ML solutions with zero human intervention.

## Documentation

For detailed usage instructions and advanced options, please refer to our tutorials:

- [Quickstart](docs/tutorials/quickstart.md)
- [LLM Providers](docs/tutorials/llm_providers.md) - Using different AI providers (Bedrock, OpenAI, Anthropic, SageMaker)
- [Interfaces](docs/tutorials/interfaces.md) - Working with different interfaces (CLI, Python API, WebUI, MCP)
- [Configuration](docs/tutorials/configuration.md) - Customizing AutoGluon Assistant settings

## 💾 Installation
For the latest features, install from source:

```bash
pip install uv && uv pip install git+https://github.com/autogluon/autogluon-assistant.git
```

*Note: If you don't have conda installed, follow conda's [official installation guide](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) to install it.*

## Quick Start

MLZero supports multiple LLM providers with AWS Bedrock as the default:

```bash
export AWS_DEFAULT_REGION="<your-region>"
export AWS_ACCESS_KEY_ID="<your-access-key>"
export AWS_SECRET_ACCESS_KEY="<your-secret-key>"
```

To run MLZero in CLI:

```bash
mlzero -i <input_data_folder>
```

## Interfaces

AutoGluon Assistant provides multiple interfaces:

### CLI

![Demo](https://github.com/autogluon/autogluon-assistant/blob/main/docs/assets/cli_demo.gif)

### Web UI

![Demo](https://github.com/autogluon/autogluon-assistant/blob/main/docs/assets/web_demo.gif)

### MCP (Model Context Protocol)

![Demo](https://github.com/autogluon/autogluon-assistant/blob/main/docs/assets/mcp_demo.gif)

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