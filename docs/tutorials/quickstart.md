# Quickstart

This quickstart guide will help you get up and running with AutoGluon Assistant (MLZero) in just a few minutes.

## Installation and Prerequisites

First, install AutoGluon Assistant:

```bash
pip install uv && uv pip install autogluon.assistant>=1.0
```

If you don't have conda installed, follow conda's [official installation guide](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) to install it.

MLZero uses AWS Bedrock by default. Configure your AWS credentials:

```bash
export AWS_DEFAULT_REGION="<your-region>"
export AWS_ACCESS_KEY_ID="<your-access-key>"
export AWS_SECRET_ACCESS_KEY="<your-secret-key>"
```

We also support Anthropic, OpenAI, and SageMaker. See our [LLM Providers](llm_providers.md) guide for details on configuring these providers.

## Basic Usage (CLI)

Here's a simple example to get you started with our command line interface:

```bash
mlzero -i <input_data_folder> [-t <optional_user_instructions>] [--provider <bedrock|openai|anthropic|sagemaker>]
```

For more detailed options and interfaces, see our [Interfaces](interfaces.md) guide.

## Next Steps

Now that you've got the basics down, explore more advanced features:

- [LLM Providers](llm_providers.md): Learn how to use different AI providers (Bedrock, OpenAI, Anthropic, SageMaker)
- [Interfaces](interfaces.md): Understand the different ways to interact with AutoGluon Assistant (CLI, Python API, WebUI, MCP)
- [Configuration](configuration.md): Master customizing AutoGluon Assistant for your specific needs

## Need Help?

If you run into any issues:

1. Check the [API Reference](../api/index.rst) for detailed documentation
2. Browse the examples for common use cases (coming soon)
3. Visit our [GitHub repository](https://github.com/autogluon/autogluon-assistant) for support
