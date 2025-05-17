#!/bin/bash

# Define base path for AutoMLAgent
AUTOML_PATH="/media/agent/AutoMLAgent"

# Function to display usage information
usage() {
    echo "Usage: $0 -training_path <path_to_training_data> -output_dir <path_to_output_folder> [-env <conda_environment>]"
    echo "Options:"
    echo " -training_path Path to the training data"
    echo " -output_dir Path to output directory"
    echo " -env Conda environment name (default: ag20250103)"
    exit 1
}

# Default values
CONDA_ENV="agent"
CONFIG_PATH="${AUTOML_PATH}/automlagent/src/automlagent/configs/default.yaml"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -training_path)
            TRAINING_PATH="$2"
            shift
            shift
            ;;
        -output_dir)
            OUTPUT_DIR="$2"
            shift
            shift
            ;;
        -env)
            CONDA_ENV="$2"
            shift
            shift
            ;;
        *)
            usage
            ;;
    esac
done

# Check if required arguments are provided
if [ -z "$TRAINING_PATH" ] || [ -z "$OUTPUT_DIR" ]; then
    usage
fi

# Extract the dataset name from the training path
DATASET_NAME=$(basename $(dirname "$TRAINING_PATH"))

# Activate conda environment
eval "$(conda shell.bash hook)"
if ! conda activate "$CONDA_ENV"; then
    echo "Failed to activate conda environment '$CONDA_ENV'"
    exit 1
fi

# Run the agent with integrated code generation and execution
python3 "${AUTOML_PATH}/run.py" \
    -i "$TRAINING_PATH" \
    -o "$OUTPUT_DIR" \
    -c "$CONFIG_PATH" \
    -n 5 \
    2>&1 | tee "${OUTPUT_DIR}/log.txt"

# Check if the process was successful
if [ $? -ne 0 ]; then
    echo "Error: Code generation and execution failed. Please check ${OUTPUT_DIR}/log.txt for details."
    conda deactivate
    exit 1
fi

echo "Process completed successfully!"
conda deactivate
echo "Results saved under ${OUTPUT_DIR}"
