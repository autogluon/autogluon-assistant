# Using Different LLM Providers

AutoGluon Assistant leverages the power of Large Language Models (LLMs) from various providers to deliver its AI capabilities. This guide explains how to configure and choose between these providers based on your specific requirements and preferences.

## Supported LLM Providers

AutoGluon Assistant currently supports the following LLM providers:

- **Amazon Bedrock** (default): This managed API service provides access to a range of foundation models from Amazon and other providers.  It is generally preferred when you seek a streamlined experience and want to avoid managing underlying infrastructure.
- **Anthropic**: Directly access Anthropic's Claude models
- **OpenAI**: Utilize the powerful GPT models offered by OpenAI
- **SageMaker**: Host and manage custom deployed models on AWS SageMaker, offering greater control and flexibility over the model and its environment. 


## Setting Up Provider Credentials

For services like OpenAI and Anthropic, you'll need to obtain and configure your API keys. For AWS services like Bedrock and SageMaker, ensure your AWS credentials and roles are correctly configured with the necessary permissions. Before using a specific provider, you need to configure the appropriate API keys or credentials:

### Amazon Bedrock (Default)

To use AWS Bedrock, set up your AWS credentials and region:

```bash
export AWS_DEFAULT_REGION="<your-region>"
export AWS_ACCESS_KEY_ID="<your-access-key>"
export AWS_SECRET_ACCESS_KEY="<your-secret-key>"
```

**Important Notes:**

- **Active AWS Account** Ensure you have an active AWS account with the necessary permissions to access and use Bedrock models.
- **Model Access**: Before you can use a specific model in Bedrock, it needs to be enabled for your account within the AWS Management Console.
- **Regional Availability**: Verify that the region you choose (e.g., us-east-1) supports the Bedrock models you intend to use. Refer to the official AWS documentation for a complete list of Bedrock supported AWS regions. Check [Bedrock supported AWS regions](https://docs.aws.amazon.com/bedrock/latest/userguide/models-regions.html) for availability.
- **SageMaker**: Make sure your IAM user or role has the appropriate permissions to interact with Bedrock services. AWS documentation provides details on managing permissions for SageMaker, which can be a good reference for similar Bedrock permission structures.

More details can be found at [Bedrock getting started](https://docs.aws.amazon.com/bedrock/latest/userguide/getting-started.html)

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

- **Performance vs. Cost**: Larger models like Claude Opus 4 or GPT-5 generally offer superior performance but come with a higher cost. Choose the model that best balances your performance needs with your budget constraints.
- **Regional Availability**: Be aware that some providers have regional restrictions for their models or specific functionalities. It's crucial to check their documentation for detailed information on regional availability before making a choice.
- **Rate Limiting**: Understand the API rate limits imposed by providers, particularly when utilizing free tiers. These limits can affect the number of requests you can make within a specific timeframe.
- **Model Updates**: Providers regularly release updates and new versions of their models. It's essential to stay informed by consulting their documentation for the latest available models and any associated changes in performance or capabilities.

## Troubleshooting

If you encounter issues with a LLM provider:

1. Verify that your credentials are correct and haven't expired.
2. Check that you have the necessary access permissions for the specific model you're trying to use.
3. Ensure the model name is properly formatted and matches the provider's documentation.
4.If applicable, verify that the chosen region supports your selected model (this is particularly relevant for services like AWS Bedrock).
5. **Inheritance Issues**: If you modify settings in the `llm` section, you must update agent references to it. The YAML anchor/alias inheritance (`<<: *default_llm`) is a one-time static reference, not dynamic. When you change the main `llm` config, agents won't automatically inherit these changes unless you explicitly update their references.