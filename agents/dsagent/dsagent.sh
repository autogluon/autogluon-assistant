#!/bin/bash

# Define base path for DS-Agent
DSAGENT_PATH="/media/agent/DS-Agent"

# Function to display usage information
usage() {
    echo "Usage: $0 -training_path <path_to_training_data> -output_dir <path_to_output_folder> [-env <conda_environment>]"
    echo "Options:"
    echo "  -training_path   Path to the training data"
    echo "  -output_dir      Path to output directory"
    echo "  -env             Conda environment name (default: dsagent)"
    exit 1
}

# Default values
CONDA_ENV="dsagent"
TIMEOUT=10800

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

# Path to the pre-generated code for this dataset
PREGENERATED_CODE="${DSAGENT_PATH}/deployment/codes/gpt-4o_True_1/${DATASET_NAME}/train_0.py"

# Check if the pre-generated code exists
if [ ! -f "$PREGENERATED_CODE" ]; then
    echo "Error: Pre-generated code for dataset '$DATASET_NAME' not found at $PREGENERATED_CODE"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Copy the pre-generated code to the output directory
OUTPUT_SCRIPT="${OUTPUT_DIR}/train.py"
cp "$PREGENERATED_CODE" "$OUTPUT_SCRIPT"

# Activate conda environment
eval "$(conda shell.bash hook)"
if ! conda activate "$CONDA_ENV"; then
    echo "Failed to activate conda environment '$CONDA_ENV'"
    exit 1
fi

echo "Running DS-Agent for dataset: $DATASET_NAME"
echo "Using pre-generated code from: $PREGENERATED_CODE"

cd ${OUTPUT_DIR}
# Run the pre-generated code
timeout $TIMEOUT python3 "$OUTPUT_SCRIPT" \
    2>&1 | tee "${OUTPUT_DIR}/log.txt"

# Check if the process was successful
if [ $? -ne 0 ]; then
    echo "Error: Execution failed. Please check ${OUTPUT_DIR}/log.txt for details."
    conda deactivate
    exit 1
fi

echo "Process completed successfully!"
conda deactivate
echo "Results saved under ${OUTPUT_DIR}"
