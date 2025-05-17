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
CONDA_ENV="codex"

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

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Activate conda environment
eval "$(conda shell.bash hook)"
if ! conda activate "$CONDA_ENV"; then
    echo "Failed to activate conda environment '$CONDA_ENV'"
    exit 1
fi

cd ${OUTPUT_DIR}

# Run the agent with integrated code generation and execution
export CODEX_UNSAFE_ALLOW_NO_SANDBOX=1
codex -q -a full-auto -m gpt-4.1 "Solve the ml task described in folder ${TRAINING_PATH}. Do not modify any files in ${TRAINING_PATH}. All temp or saved files should be located somewhere under ${OUTPUT_DIR}. Save the predicted results in the same format as training data to ${OUTPUT_DIR}. Name the result file 'results.xxx', where the extension should be same as the test data file." 2>&1 | tee "${OUTPUT_DIR}/log.txt"

# Check if the process was successful
if [ $? -ne 0 ]; then
    echo "Error: Code generation and execution failed. Please check ${OUTPUT_DIR}/log.txt for details."
    conda deactivate
    exit 1
fi

echo "Process completed successfully!"
conda deactivate
echo "Results saved under ${OUTPUT_DIR}"
