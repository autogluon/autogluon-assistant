# AutoML Agent

This repository contains an ML agent that generates and executes code based on input data and configuration settings. The agent can work with various machine learning tools and frameworks while allowing for optional interactive user input during the code generation process.

## Prerequisites

- Python 3.9/10/11/12
- Conda package manager
- AutoGluon dependencies
- Access to Bedrock/OpenAI API

## Setup

1. Clone the repository:
```bash
git clone https://github.com/FANGAreNotGnu/AutoMLAgent.git
cd AutoMLAgent
```

2. Install the package:
```bash
conda create -n agent python=3.11 -y
conda activate agent
pip install -e .
```

2.1 Install Object Detection Dependencies
```bash
pip install -U pip setuptools wheel
sudo apt-get install -y ninja-build gcc g++
python3 -m mim install "mmcv==2.1.0"
python3 -m pip install "mmdet==3.2.0"
python3 -m pip install "mmengine>=0.10.6"
```

3. Configure your environment variables:
```bash
# Similar to Autogluon Assistant
export OPENAI_API_KEY=your_api_key
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=your_region
```

## Usage

### Command Line Interface

The main script `run.py` provides a command-line interface with the following options:

```bash
python run.py -i INPUT_DATA_FOLDER -o OUTPUT_DIR -c CONFIG_PATH [-n MAX_ITERATIONS] [--need_user_input]
```

Arguments:
- `-i, --input_data_folder`: Path to the folder containing input data (required)
- `-o, --output_dir`: Path to the output directory for generated files (required)
- `-c, --config_path`: Path to the configuration file (required)
- `-n, --max_iterations`: Maximum number of iterations for code generation (default: 5)
- `--need_user_input`: Enable user input between iterations (optional flag)

Example:
```bash
python run.py -i ./data -o ./output -c config.yaml -n 3
```

### Adding Third-Party ML Tools

To interactively add new machine learning tools to the agent:

```bash
python3 tools/add_tools.py
```

This will guide you through the process of integrating additional ML frameworks or libraries.

## Project Structure

```
AutoMLAgent/
├── LICENSE
├── README.md
├── automlagent/
│   └── src/
│       └── automlagent/
│           ├── __init__.py
│           ├── coder/
│           │   ├── __init__.py
│           │   ├── llm_coder.py
│           │   └── utils.py
│           ├── coding_agent.py
│           ├── configs/
│           │   ├── agrag/
│           │   │   ├── agrag_object_detection.yaml
│           │   │   └── agrag_semantic_segmentation.yaml
│           │   └── default.yaml
│           ├── constants.py
│           ├── llm/
│           │   ├── __init__.py
│           │   └── llm_factory.py
│           ├── prompt/
│           │   ├── __init__.py
│           │   ├── data_prompt.py
│           │   ├── error_prompt.py
│           │   ├── execution_prompt.py
│           │   ├── prompt_aggregation.py
│           │   ├── task_prompt.py
│           │   ├── tutorial_prompt.py
│           │   ├── user_prompt.py
│           │   └── utils.py
│           └── tools_registry/
│               ├── __init__.py
│               ├── _common/
│               ├── autogluon.multimodal/
│               ├── autogluon.tabular/
│               ├── autogluon.timeseries/
│               └── registry.py
├── run.py
├── setup.py
└── tools/
    ├── add_tools.py
    └── convert_notebooks.py
```

## Output Files

The agent generates a structured output directory with the following organization:

```
outputs/
├── description_analysis.txt     # Analysis of the problem description
├── description_files.txt       # List of input files analyzed
├── error_analysis.json         # Error tracking and analysis
├── eval_log.txt               # Evaluation metrics and logs
├── log.txt                    # General execution log
├── results.csv                # Final results and predictions
├── selected_tutorials.json     # Selected tutorials for the task
├── task_description.txt       # Original task description
├── tool_selection.txt         # Selected ML tools and reasoning
├── tutorial_prompt.txt        # Generated tutorial prompt
├── tutorial_contents/         # Retrieved tutorials
│   ├── tutorial_1.md
│   ├── tutorial_2.md
│   └── tutorial_3.md
├── iteration_0/              # First iteration
│   ├── coding_prompt.txt     # Prompt for code generation
│   ├── execution_prompt.txt  # Prompt for execution
│   ├── execution_script.sh   # Generated shell script
│   ├── generated_code.py     # Generated Python code
│   └── states/              # Iteration state information
│       ├── bash_script.sh
│       ├── data_prompt.txt
│       ├── error_message.txt
│       ├── python_code.py
│       ├── stderr.txt
│       ├── stdout.txt
│       ├── task_prompt.txt
│       ├── tutorial_prompt.txt
│       └── user_input.txt
├── iteration_1/             # Second iteration
│   └── ...                 # Same structure as iteration_0
├── iteration_2/             # Third iteration
│   └── ...                 # Same structure as iteration_0
├── iteration_3/             # Fourth iteration
│   └── ...                 # Same structure as iteration_0
└── model_YYYYMMDD_HHMMSS/   # Trained model directory
```

Each iteration directory contains the prompts, generated code, and execution states for that specific iteration. The model directory contains all artifacts related to the trained model, including individual model components, logs, and utilities.
Multiple model directories may be created with timestamps (YYYYMMDD_HHMMSS) during different stages of the training process. Each contains its own complete set of model artifacts.

## Configuration

A YAML configuration file to control:
- Tutorial generation parameters
- LLM provider settings (Bedrock or OpenAI)
- Model selection and parameters
- Agent-specific configurations

A default configuration is provided at `AutoMLAgent/automlagent/src/automlagent/configs/default.yaml`:
```yaml
# Tutorial Prompt Generator Configuration
max_chars_per_file: 100
max_num_tutorials: 3
max_user_input_length: 9999
max_error_message_length: 9999
max_tutorial_length: 99999
create_venv: false
condense_tutorials: false

# Default LLM Configuration
# For each agent (coder, etc.) you can use a different one
llm: &default_llm
  # Note: bedrock is only supported in limited AWS regions
  # and requires AWS credentials
  provider: bedrock
  model: "anthropic.claude-3-5-sonnet-20241022-v2:0"
  # Alternative configuration:
  # provider: openai
  # model: gpt-4-0314
  max_tokens: 4096
  proxy_url: null
  temperature: 0
  verbose: True
  multi_turn: True

coder:
  <<: *default_llm  # Merge llm_config
  temperature: 0.5
  max_tokens: 4096
  top_p: 1
  multi_turn: True
```

