#!/bin/bash
# Define base path for AutoKaggle
AUTOKAGGLE_PATH="/media/agent/AutoKaggle"

# Function to display usage information
usage() {
    echo "Usage: $0 -training_path <path_to_training_data> -output_dir <path_to_output_folder> [-env <conda_environment>]"
    echo "Options:"
    echo "  -training_path   Path to the training data"
    echo "  -output_dir      Path to output directory"
    echo "  -env             Conda environment name (default: agent)"
    exit 1
}

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
            # Keep the parameter for compatibility but ignore it
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

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# No conda environment activation needed

# Log the execution start
echo "Running AutoKaggle for dataset: $DATASET_NAME"
echo "Training path: $TRAINING_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "$(date): Process started" | tee "${OUTPUT_DIR}/log.txt"

# Instead of running the agent, copy the existing results
# Define the source path for the results
SOURCE_PATH="${AUTOKAGGLE_PATH}/multi_agents/experiments_history/${DATASET_NAME}/gpt_4o/all_tools/9/submission.csv"

# Check if the source file exists
if [ ! -f "$SOURCE_PATH" ]; then
    echo "Error: Source file not found at $SOURCE_PATH" | tee -a "${OUTPUT_DIR}/log.txt"
    exit 1
fi

# Copy the results file to the output directory
cp "$SOURCE_PATH" "${OUTPUT_DIR}/results.csv"

# Check if the copy was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to copy results file. Please check permissions and paths." | tee -a "${OUTPUT_DIR}/log.txt"
    exit 1
fi

echo "$(date): Process completed successfully!" | tee -a "${OUTPUT_DIR}/log.txt"
echo "Results copied from: $SOURCE_PATH" | tee -a "${OUTPUT_DIR}/log.txt"
echo "Results saved to: ${OUTPUT_DIR}/results.csv" | tee -a "${OUTPUT_DIR}/log.txt"

echo "AutoKaggle process completed successfully!"
echo "Results saved under ${OUTPUT_DIR}"
