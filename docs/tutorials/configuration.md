# Configuration Customization

AutoGluon Assistant uses YAML configuration files to control its behavior. This tutorial explains the configuration system and how to customize it for your specific needs.

## Configuration Overview

The configuration system is based on hierarchical YAML files that control:

1. General execution settings
2. LLM provider settings
3. Agent behaviors and parameters
4. Resource utilization
5. Data handling preferences

## Basic Structure

A configuration file has this general structure:

```yaml
# General settings
per_execution_timeout: 86400
create_venv: false
# ... other general settings

# Default LLM Configuration
llm: &default_llm  # The anchor defines reusable settings
  provider: bedrock
  model: "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
  # ... other LLM settings

# Agent-specific configurations
coder:
  <<: *default_llm  # This merges all settings from default_llm
  multi_turn: True  # Override specific settings
```

## Configuration Parameters

### General Settings

| Parameter | Description | Default |
|-----------|-------------|---------|
| `per_execution_timeout` | Maximum execution time (seconds) for code execution | 86400 |
| `create_venv` | Whether to install additional packages in created conda environment | false |
| `condense_tutorials` | Whether to use condensed tutorials | true |
| `use_tutorial_summary` | Whether to use tutorial summary as retrieval key | true |
| `continuous_improvement` | Continue iterations after finding a valid solution | false |
| `optimize_system_resources` | Optimize resource usage during execution | false |
| `cleanup_unused_env` | Remove unused environments after execution | true |

### Data Perception Settings

| Parameter | Description | Default |
|-----------|-------------|---------|
| `max_file_group_size_to_show` | Minimum number of similar files to show as a group | 5 |
| `num_example_files_to_show` | Number of example files to display for each type | 1 |
| `max_chars_per_file` | Maximum characters to display per file | 768 |
| `max_user_input_length` | Maximum length of user input to process | 2048 |
| `max_error_message_length` | Maximum length of error messages to include | 2048 |

### Tutorial Settings

| Parameter | Description | Default |
|-----------|-------------|---------|
| `num_tutorial_retrievals` | Number of tutorial segments to retrieve | 30 |
| `max_num_tutorials` | Maximum number of tutorials to include | 5 |
| `max_tutorial_length` | Maximum length of all tutorial contents | 32768 |

### LLM Settings

| Parameter | Description | Default |
|-----------|-------------|---------|
| `provider` | LLM provider to use (bedrock, openai, anthropic, sagemaker) | bedrock |
| `model` | Specific model name for the selected provider | <provider-specific> |
| `max_tokens` | Maximum token limit for model responses | 65535 |
| `proxy_url` | Optional proxy URL for API requests | null |
| `temperature` | Controls randomness (0.0-1.0, lower = more deterministic) | 0.1 |
| `top_p` | Nucleus sampling parameter for token selection | 0.9 |
| `verbose` | Whether to log detailed information about LLM interactions | true |
| `multi_turn` | Whether to use multi-turn conversation with the LLM across different iterations | false |
| `template` | Optional custom prompt template | null |
| `add_coding_format_instruction` | Add explicit coding format instructions | false |

## Agent-Specific Settings

AutoGluon Assistant uses specialized agents for different tasks. Each inherits the default LLM settings but can have custom overrides:

### Coder Agent

Generates code based on requirements and context.

```yaml
coder:
  <<: *default_llm
  multi_turn: True  # Enable multi-turn conversation across iterations for iterative coding
```

### Executer Agent

Runs code and evaluates execution results.

```yaml
executer:
  <<: *default_llm
  max_stdout_length: 8192  # Maximum length of stdout to capture
  max_stderr_length: 2048  # Maximum length of stderr to capture
```

### Reader Agent

Analyzes and understands input data files.

```yaml
reader:
  <<: *default_llm
  details: False  # Whether to include detailed file information
```

### Task Descriptor Agent

Describes tasks based on input data.

```yaml
task_descriptor:
  <<: *default_llm
  max_description_files_length_to_show: 1024         # Max length to show
  max_description_files_length_for_summarization: 16384  # Max length for summarization
```

### Other Specialized Agents

- `error_analyzer`: Analyzes execution errors and suggests fixes
- `retriever`: Retrieves relevant information from tutorials
- `reranker`: Re-ranks retrieved information for relevance
- `description_file_retriever`: Retrieves information from description files
- `tool_selector`: Selects appropriate tools based on requirements

## Use a Custom Configuration

You can create and use a custom configuration file by:

1. Create a new YAML file, e.g., `my_custom_config.yaml`
2. Run with your custom config: `mlzero -i <input_folder> -c my_custom_config.yaml`

## Best Practices

1. **Start Simple**: Begin with minimal customizations and add more as needed
2. **Test Incrementally**: Test changes one at a time to understand their impact
3. **Document Your Configs**: Add comments to explain your specific changes
4. **Version Control**: Keep your configurations in version control
5. **Use Environment Variables**: Store sensitive information in environment variables

## Troubleshooting

- **Validation Errors**: Check for YAML syntax errors and invalid parameters
- **Inconsistent Behavior**: Ensure all necessary settings are properly overridden
- **Provider Issues**: Verify API keys and model availability for your chosen provider
