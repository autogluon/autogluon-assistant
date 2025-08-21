# Agent Customization

This tutorial demonstrates how to customize AutoGluon Assistant agents for specific domains using a data visualization example.

## Customizing Agents with Templates

Customize agent behavior by modifying prompt templates using {variable_name} syntax, through either:

1. **External Template Files**: Write the template in a text file and use its absolute path in your configuration file
2. **Inline Templates**: Write the template directly in your configuration file

## Example: Data Visualization

Here we give an example of customizing the system to complete data visualization tasks using inline templates. Create `data_visualizer.yaml` with customized templates:

```yaml
# data_visualizer.yaml (key sections only)

python_coder:
  <<: *default_llm
  multi_turn: True
  template: |
    As a Data Visualization Agent, you will generate Python code using {selected_tool} to create insightful visualizations. Follow these specifications:
    
    ONLY save files to: {per_iteration_output_folder}
    
    1. Data preprocessing:
       - Clean the data (handle missing values, outliers)
       - Format data for visualization
    
    2. Visualization creation:
       - Create visualizations based on data type and requirements
       - For multivariate data: correlation plots, pair plots
       - For time-series: line plots, seasonal decomposition
       - For categorical: bar charts, heatmaps
       - For geographic: maps with appropriate projections
    
    3. Output organization:
       - Save in appropriate formats (PNG, SVG, HTML)
       - Create a main script that generates all visualizations
    
    4. Documentation:
       - Include docstrings and comments
       - Document visualization choices and insights
    
    ### Task Description
    {task_description}
    
    ### Data Structure
    {data_prompt}
    
    ### User Instruction
    {user_input_truncate_end_2048}

task_descriptor:
  <<: *default_llm
  template: |
    Provide a precise description of the data visualization task focusing on:
    
    1. Main visualization objectives (exploration, explanation, comparison)
    2. Key variables to visualize
    3. Specific visualization types required
    4. Target audience and required insights
    
    ### Data Structure:
    {data_prompt}
    
    ### Description File Contents:
    {description_file_contents_truncate_end_16384}

tool_selector:
  <<: *default_llm
  template: |
    Select the most appropriate library for this visualization task.
    
    ### Task Description:
    {task_description}
    
    ### Data Information:
    {data_prompt}
    
    ### Available Libraries:
    {tools_info}
    
    Consider visualization types needed (static/interactive), data complexity, and output requirements.
    
    IMPORTANT: Format your response as:
    ---
    SELECTED_LIBRARY: <library_name>
    EXPLANATION: <reasoning>
    ---
```

### Key Changes

1. **PythonCoderAgent**: Ask to create effective visualizations (instead of previous request for model training and inference)
2. **TaskDescriptorAgent**: Identifies visualization objectives (instead of previous ML task description for model training and inference)
3. **ToolSelectorAgent**: Evaluates libraries based on visualization requirements (instead of previous request for model training and inference)

### Usage

Run your customized agent:

```bash
mlzero -i </path/to/data> -t "visualize the training data distribution, and generate a .pdf report" -c </path/to/data_visualizer.yaml>
```

```

### Example Output on Abalone Dataset

[Abalone Data Visualization Report (PDF)](../assets/abalone_data_visualization_report.pdf)

The visualization agent creates a comprehensive report with distribution plots, correlation analyses, and dimension reduction visualizations.

## Template Variables

AutoGluon Assistant uses a variable registration system for template rendering. These variables can be used in your custom templates (they will be filled during the run using the information gathered by LLMs):

### User and Task Variables
- `user_input`: Raw user instructions and requirements
- `task_description`: Description of the task to be performed

### Data-Related Variables
- `data_prompt`: Information about data structure and files 
- `description_file_contents`: Contents of identified description files

### Path and Output Variables
- `per_iteration_output_folder`: Output directory for current iteration
- `python_file_path`: Path to the Python file to execute
- `file_path`: Path to the file being read
- `file_size_mb`: Size of the file in MB
- `max_chars`: Maximum characters to read

### Code-Related Variables
- `python_code`: Python code of current iteration
- `previous_python_code`: Python code of previous iteration
- `bash_script`: Bash script of current iteration
- `previous_bash_script`: Bash script of previous iteration

### Error and Debugging Variables
- `previous_error_message`: Error message from previous iteration
- `all_error_analyses`: Error analyses from all completed iterations
- `stdout`: Standard output from code execution
- `stderr`: Standard error from code execution

### Tools and Tutorials Variables
- `selected_tool`: The chosen ML library/framework
- `tool_prompt`: Information about selected tools
- `tutorial_prompt`: Relevant tutorial information
- `previous_tutorial_prompt`: Tutorial information from previous iteration
- `max_num_tutorials`: Maximum number of tutorials to select

### Environment and Validation Variables
- `environment_prompt`: System environment information
- `validation_prompt`: Asking for Validation
- `best_code_prompt`: Asking to improve the code

### Variable Truncation

For long variable content, use these truncation modifiers:

- `{variable_name_truncate_start_N}`: Truncate from start, keeping last N characters
  Example: `{user_input_truncate_start_1000}` keeps the last 1000 characters
  
- `{variable_name_truncate_end_N}`: Truncate from end, keeping first N characters
  Example: `{description_file_contents_truncate_end_2048}` keeps the first 2048 characters
  
- `{variable_name_truncate_mid_N}`: Truncate from middle, keeping N/2 characters from start and end
  Example: `{python_code_truncate_mid_3000}` keeps 1500 characters from start and end

## Best Practices

- Modify from the existing default prompt
- Start by customizing the most critical agent (e.g. PythonCoderAgent and TaskDescriptorAgent)
- Embed domain-specific guidelines and specify clear output requirements and formats

Creating domain-specific agents through prompt engineering allows you to build specialized tools that excel at specific tasks while leveraging AutoGluon Assistant's architecture.
