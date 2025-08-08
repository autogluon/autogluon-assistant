# Configuration Customization

AutoGluon Assistant uses YAML configuration files to control its behavior. This tutorial explains how to customize these configurations for your specific needs.

## Configuration Overview

The configuration system is based on hierarchical YAML files that control:

1. LLM provider settings
2. Agent behaviors and parameters
3. Resource utilization
4. Data handling preferences
5. Runtime parameters

## Default Configuration Structure

The default configuration (`default.yaml`) serves as the base for all other configurations:

```yaml
# Basic settings
per_execution_timeout: 86400
max_file_group_size_to_show: 5
num_example_files_to_show: 1
max_chars_per_file: 768
num_tutorial_retrievals: 30
max_num_tutorials: 5
max_user_input_length: 2048
max_error_message_length: 2048
max_tutorial_length: 32768
create_venv: false
condense_tutorials: True
use_tutorial_summary: True
continuous_improvement: False
optimize_system_resources: False
cleanup_unused_env: True

# Default LLM Configuration
llm: &default_llm
  provider: bedrock
  model: "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
  max_tokens: 65535
  proxy_url: null
  temperature: 0.1
  top_p: 0.9
  verbose: True
  multi_turn: False
  template: null
  add_coding_format_instruction: false

# Agent-specific configurations
coder:
  <<: *default_llm  # Inherit from default_llm
  multi_turn: True

executer:
  <<: *default_llm  # Inherit from default_llm
  max_stdout_length: 8192
  max_stderr_length: 2048

reader:
  <<: *default_llm  # Inherit from default_llm
  details: False

# ...other agents
```

## Creating a Custom Configuration

You can create a custom configuration file to override specific settings:

1. Create a new YAML file, e.g., `my_custom_config.yaml`
2. Include only the settings you want to override
3. Run with your custom config: `mlzero -i <input_folder> -c my_custom_config.yaml`

Example custom configuration:

```yaml
# Customize LLM settings
llm: &default_llm
  provider: anthropic
  model: "claude-3-opus-20240229"
  temperature: 0.2
  max_tokens: 100000

# Customize specific agent
coder:
  <<: *default_llm
  temperature: 0.3
  multi_turn: True

# Override runtime settings
continuous_improvement: True
max_file_group_size_to_show: 10
```

## Key Configuration Parameters

### General Settings

| Parameter | Description | Default |
|-----------|-------------|---------|
| `per_execution_timeout` | Maximum execution time in seconds | 86400 (24 hours) |
| `create_venv` | Whether to create virtual environments | false |
| `continuous_improvement` | Continue optimizing after success | false |
| `optimize_system_resources` | Optimize resource usage | false |

### LLM Settings

| Parameter | Description | Default |
|-----------|-------------|---------|
| `provider` | LLM provider (bedrock, openai, anthropic, sagemaker) | bedrock |
| `model` | Specific model name | provider-dependent |
| `max_tokens` | Maximum token limit | 65535 |
| `temperature` | Model temperature (randomness) | 0.1 |
| `top_p` | Nucleus sampling parameter | 0.9 |
| `proxy_url` | Optional proxy URL | null |

### Agent-Specific Settings

Each agent can have custom settings that override the default LLM configuration:

```yaml
coder:
  <<: *default_llm  # Inherit all default settings
  temperature: 0.3  # Override specific parameters
  multi_turn: True
```

Available agents:
- `coder`: Generates code
- `executer`: Runs code and evaluates results
- `reader`: Analyzes and understands input data
- `error_analyzer`: Analyzes errors in execution
- `retriever`: Retrieves relevant information
- `reranker`: Re-ranks retrieved information
- `task_descriptor`: Describes tasks
- `tool_selector`: Selects appropriate tools

## Advanced Configuration Techniques

### YAML Anchors and Aliases

YAML anchors (`&`) and aliases (`*`) allow you to define a set of parameters once and reuse them:

```yaml
# Define common settings
common: &common_settings
  temperature: 0.2
  top_p: 0.9
  verbose: True

# Use common settings with specific overrides
llm:
  <<: *common_settings  # Include all common settings
  provider: openai
  model: "gpt-4o-2024-08-06"

coder:
  <<: *common_settings
  temperature: 0.3  # Override a specific setting
```

### Environment Variable Expansion

Some configurations support environment variable expansion:

```yaml
llm:
  provider: anthropic
  api_key: ${ANTHROPIC_API_KEY}  # Will be replaced with env var value
```

## Configuration Examples

### High-Performance Configuration

```yaml
# Configuration for high-performance tasks
llm: &default_llm
  provider: bedrock
  model: "us.anthropic.claude-3-opus-20240229-v1:0"
  temperature: 0.1
  max_tokens: 100000

optimize_system_resources: True
continuous_improvement: True
max_iterations: 10
```

### Fast-Iteration Configuration

```yaml
# Configuration for quick iterations
llm: &default_llm
  provider: anthropic
  model: "claude-3-haiku-20240307"
  temperature: 0.2
  max_tokens: 32768

optimize_system_resources: True
continuous_improvement: False
max_iterations: 3
```

### Multi-Provider Configuration

```yaml
# Define different providers for different agents
llm: &default_llm
  provider: bedrock
  model: "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
  temperature: 0.1

coder:
  <<: *default_llm
  
executer:
  provider: anthropic
  model: "claude-3-sonnet-20240229"
  temperature: 0.1
  
reader:
  provider: openai
  model: "gpt-4o-2024-08-06"
  temperature: 0.1
```

## Best Practices

1. **Start Simple**: Begin with minimal customizations and add more as needed
2. **Test Incrementally**: Test changes one at a time to understand their impact
3. **Document Your Configs**: Add comments to explain why you've made specific changes
4. **Version Control**: Keep your configurations in version control
5. **Use Environment Variables**: Store sensitive information like API keys in environment variables

## Common Configuration Scenarios

### Adjusting Model Creativity

```yaml
# For more creative solutions
llm:
  temperature: 0.5  # Higher temperature for more randomness
  top_p: 0.95       # Broader token selection
```

### Managing Resource Usage

```yaml
# For resource-constrained environments
optimize_system_resources: True
cleanup_unused_env: True
create_venv: False
```

### Debugging Configuration

```yaml
# For troubleshooting
llm:
  verbose: True
  
# Increase logging detail  
# (Can be set via CLI with -v 4)
verbosity: 4
```

## Troubleshooting Configuration Issues

- **Validation Errors**: Check for YAML syntax errors and invalid parameter values
- **Inconsistent Behavior**: Ensure all necessary settings are properly overridden
- **Provider Issues**: Verify API keys and model availability for your provider
- **File Paths**: Use absolute paths when referencing external files

When in doubt, try running with the default configuration first, then gradually add your customizations.