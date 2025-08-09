# Using Different LLM Providers

AutoGluon Assistant supports multiple LLM providers to power its AI capabilities. This tutorial explains how to configure and use different providers based on your preferences and requirements.

## Supported LLM Providers

AutoGluon Assistant currently supports the following LLM providers:

- **AWS Bedrock** (default): Managed API service for foundation models from Amazon
- **Anthropic**: Claude models directly from Anthropic
- **OpenAI**: GPT models from OpenAI
- **SageMaker**: Custom deployed models on AWS SageMaker

## Setting Up Provider Credentials

Before using a specific provider, you need to configure the appropriate API keys or credentials:

### AWS Bedrock (Default)

To use AWS Bedrock, set up your AWS credentials and region:

```bash
export AWS_DEFAULT_REGION="<your-region>"
export AWS_ACCESS_KEY_ID="<your-access-key>"
export AWS_SECRET_ACCESS_KEY="<your-secret-key>"
```

Make sure you have an active AWS account with access to Bedrock models in your specified region. Check [Bedrock supported AWS regions](https://docs.aws.amazon.com/bedrock/latest/userguide/models-regions.html) for availability.

### Anthropic

To use Anthropic's Claude models directly, set your API key:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

You can create an Anthropic account [here](https://console.anthropic.com/) and manage your API keys in the [Console](https://console.anthropic.com/keys).

### OpenAI

For OpenAI models, set your API key:

```bash
export OPENAI_API_KEY="sk-..."
```

You can sign up for an OpenAI account [here](https://platform.openai.com/) and manage your API keys [here](https://platform.openai.com/account/api-keys).

### SageMaker

For custom models deployed on SageMaker, configure:

```bash
# Basic AWS credentials (same as for Bedrock)
export AWS_DEFAULT_REGION="<your-region>"
export AWS_ACCESS_KEY_ID="<your-access-key>"
export AWS_SECRET_ACCESS_KEY="<your-secret-key>"
```

## Selecting a Provider via CLI

The easiest way to select a provider is through the CLI using the `--provider` option:

```bash
# Use Bedrock (default)
mlzero -i <input_data_folder> --provider bedrock

# Use OpenAI
mlzero -i <input_data_folder> --provider openai

# Use Anthropic
mlzero -i <input_data_folder> --provider anthropic

# Use SageMaker
mlzero -i <input_data_folder> --provider sagemaker
```

This option will automatically use the appropriate configuration file for your selected provider.

## Using Provider-Specific Configuration Files

Each provider has a dedicated configuration file:

- `bedrock.yaml` (default provider)
- `openai.yaml`
- `anthropic.yaml`
- `sagemaker.yaml`

You can directly specify a provider's config file:

```bash
mlzero -i <input_data_folder> -c <path_to_configs>/openai.yaml
```

## Custom Configuration

You can create a custom configuration based on any provider's template:

1. Copy the provider-specific YAML file:
   ```bash
   cp <path_to_configs>/bedrock.yaml my_custom_config.yaml
   ```

2. Modify the provider and model settings:
   ```yaml
   llm: &default_llm
     provider: anthropic  # Change to your preferred provider
     model: "claude-3-7-sonnet-20250219"  # Change to your preferred model
     # ... other settings
   ```

3. Use your custom config:
   ```bash
   mlzero -i <input_data_folder> -c my_custom_config.yaml
   ```


## Best Practices

- **Performance vs. Cost**: Larger models like Claude-4-Opus or GPT-5 offer better performance but cost more. Choose based on your needs.
- **Regional Availability**: Some providers have regional restrictions. Check their documentation for details.
- **Rate Limiting**: Be aware of provider rate limits, especially on free tiers.
- **Model Updates**: Providers regularly update their models. Check their documentation for the latest available models.

## Troubleshooting

If you encounter issues with a provider:

1. Verify your credentials are correct and not expired
2. Check that you have access to the specific model
3. Ensure you've properly formatted the model name
4. Verify the region supports your chosen model (for AWS services)
5. **Inheritance Issues**: If you modify settings in the `llm` section, you must update agent references to it. The YAML anchor/alias inheritance (`<<: *default_llm`) is a one-time static reference, not dynamic. When you change the main `llm` config, agents won't automatically inherit these changes unless you explicitly update their references.