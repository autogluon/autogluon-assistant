---
sd_hide_title: true
hide-toc: true
---

# AutoGluon-Assistant

::::::{div} landing-title
:style: "padding: 0.1rem 0.5rem 0.6rem 0; background-image: linear-gradient(315deg, #438ff9 0%, #3977B9 74%); clip-path: polygon(0px 0px, 100% 0%, 100% 100%, 0% calc(100% - 1.5rem)); -webkit-clip-path: polygon(0px 0px, 100% 0%, 100% 100%, 0% calc(100% - 1.5rem));"

::::{grid}
:reverse:
:gutter: 2 3 3 3
:margin: 4 4 1 2

:::{grid-item}
:columns: 12 4 4 4

```{image} ./_static/autogluon-s.png
:width: 200px
:class: sd-m-auto sd-animate-grow50-rot20
```
:::

:::{grid-item}
:columns: 12 8 8 8
:child-align: justify
:class: sd-text-white sd-fs-3

AutoGluon-Assistant: AI-Powered Assistant for AutoML and Beyond
:::
::::
::::::

---

AutoGluon-Assistant is an AI-powered assistant that helps users with AutoML tasks and provides intelligent guidance for machine learning automation. It combines the power of AutoGluon's automated machine learning capabilities with conversational AI to make machine learning more accessible and efficient.

## Key Features

- **Intelligent ML Guidance**: Get expert advice on ML/DL workflows and best practices
- **Chat Mode**: Interactive Q&A for conversational guidance without code execution
- **Multiple Interfaces**: Use through CLI, Python API, WebUI, or MCP integration
- **Flexible LLM Providers**: Choose from AWS Bedrock, Anthropic, OpenAI, or SageMaker
- **Code Generation**: Generate code for any registered machine learning or deep learning framework (for unregistered frameworks, performance may be reduced due to limited framework-specific knowledge)
- **Multi-modal Support**: Handle various data types including tabular, text, image, and multimodal data
- **Customizable Configuration**: Extensive configuration options for tailoring to your needs

## Quick Start

### CLI Interface

```bash
# Using CLI
mlzero -i <input_data_folder> -t "Train a classification model for this dataset" --provider bedrock
```

### Python API

```python
# Using Python API
from autogluon.assistant import run_agent

run_agent(
    input_data_folder="./my_data",
    initial_user_input="Train a classification model for image data using PyTorch"
)
```

## Documentation Structure

```{toctree}
:maxdepth: 2
:hidden:

tutorials/index
api/index
whats_new/index
```

::::{grid} 1 2 2 3
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Tutorials
:link: tutorials/index
:link-type: doc

Step-by-step guides to get you started with AutoGluon-Assistant
:::

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` API Reference
:link: api/index
:link-type: doc

Complete API documentation for all AutoGluon-Assistant modules
:::

:::{grid-item-card} {octicon}`megaphone;1.5em;sd-mr-1` What's New
:link: whats_new/index
:link-type: doc

Latest updates, features, and improvements
:::
::::

## Installation

Install AutoGluon-Assistant using pip:

```bash
pip install autogluon.assistant
```

For development installation:

```bash
git clone https://github.com/autogluon/autogluon-assistant.git
cd autogluon-assistant
pip install -e .
```

## Community

- **GitHub**: [autogluon/autogluon-assistant](https://github.com/autogluon/autogluon-assistant)
- **Issues**: Report bugs and request features on [GitHub Issues](https://github.com/autogluon/autogluon-assistant/issues)
- **Discussions**: Join the community on [GitHub Discussions](https://github.com/autogluon/autogluon-assistant/discussions)

## License

AutoGluon-Assistant is licensed under the [Apache 2.0 License](https://github.com/autogluon/autogluon-assistant/blob/main/LICENSE).
