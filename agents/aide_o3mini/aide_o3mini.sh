#!/bin/bash
# Function to display usage information
usage() {
    echo "Usage: $0 -training_path <path_to_training_data> -output_dir <path_to_output_folder> [-env <conda_environment>]"
    echo "Options:"
    echo "  -training_path Path to the training data"
    echo "  -output_dir Path to output directory"
    echo "  -env Conda environment name (default: aide)"
    exit 1
}
# Default values
CONDA_ENV="aide"
AIDE_WORKING_DIR="/media/agent/aideml/workspaces"
AIDE_ADDITIONAL_PROMPT="$(dirname "$0")/additional_prompt.txt"
MODEL="o3-mini"
STEPS=5
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
# Generate AIDE_WORKING_NAME with timestamp and UUID
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
UUID=$(cat /dev/urandom | tr -dc 'a-f0-9' | fold -w 4 | head -n 1)
AIDE_WORKING_NAME="${DATASET_NAME}_${TIMESTAMP}_${UUID}"

# Create temp file with concatenated descriptions and additional prompt
TEMP_UUID=$(cat /dev/urandom | tr -dc 'a-f0-9' | fold -w 8 | head -n 1)
TEMP_DESC_FILE="${OUTPUT_DIR}/combined_description_${TEMP_UUID}.txt"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Concatenate the description and additional prompt files
cat "$TRAINING_PATH/descriptions.txt" "$AIDE_ADDITIONAL_PROMPT" > "$TEMP_DESC_FILE"

# Activate conda environment
eval "$(conda shell.bash hook)"
if ! conda activate "$CONDA_ENV"; then
    echo "Failed to activate conda environment '$CONDA_ENV'"
    exit 1
fi
# Run the agent with integrated code generation and execution
timeout $TIMEOUT aide data_dir=$TRAINING_PATH \
    desc_file=$TEMP_DESC_FILE \
    exp_name=$AIDE_WORKING_NAME \
    workspace_dir=$AIDE_WORKING_DIR \
    agent.steps=$STEPS \
    agent.code.model=$MODEL \
    agent.feedback.model=$MODEL \
    preprocess_data=False \
    copy_data=False \
    2>&1 | tee "${OUTPUT_DIR}/log.txt"
# Find the AIDE working directory and copy results
AIDE_RESULT_DIR=$(find "$AIDE_WORKING_DIR" -type d -name "*${AIDE_WORKING_NAME}" -print -quit)
if [ -z "$AIDE_RESULT_DIR" ]; then
    echo "Error: Could not find AIDE working directory for ${AIDE_WORKING_NAME}"
    conda deactivate
    rm -f "$TEMP_DESC_FILE"  # Clean up temp file
    exit 1
fi
# Copy the results file
if [ -f "${AIDE_RESULT_DIR}/working/submission.csv" ]; then
    cp "${AIDE_RESULT_DIR}/working/submission.csv" "${OUTPUT_DIR}/results.csv"
else
    echo "Error: Could not find submission.csv in ${AIDE_RESULT_DIR}/working/"
    conda deactivate
    rm -f "$TEMP_DESC_FILE"  # Clean up temp file
    exit 1
fi
# Check if the process was successful
if [ $? -ne 0 ]; then
    echo "Error: Code generation and execution failed. Please check ${OUTPUT_DIR}/log.txt for details."
    conda deactivate
    rm -f "$TEMP_DESC_FILE"  # Clean up temp file
    exit 1
fi
echo "Process completed successfully!"
conda deactivate
echo "Results saved under ${OUTPUT_DIR}"
