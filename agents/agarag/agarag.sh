#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: $0 -training_path <path_to_training_data> -output_dir <path_to_output_folder>"
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

eval "$(conda shell.bash hook)"
conda activate ag_agrag_20241120 || { echo "Failed to activate conda environment 'ag_agrag_20241120'"; exit 1; }
python3 /media/agent/AutoMLAgent/run_agent.py \
    -i "$TRAINING_PATH" \
    -w "$OUTPUT_DIR" \
    -m anthropic.claude-3-5-sonnet-20241022-v2:0 \
    -b "agrag" \
    -l "https://auto.gluon.ai/stable/tutorials/multimodal/image_segmentation/index.html" \
    2>&1 | tee "${OUTPUT_DIR}/log.txt"

# Run the generated code
echo "Running the generated code..."
python "${OUTPUT_DIR}/generated_code.py" 2>&1 | tee "${OUTPUT_DIR}/generated_code_log.txt"
if [ $? -ne 0 ]; then
    echo "Error: Failed to run the generated code. Please check the output file for any issues."
    exit 1
fi
echo "Process completed successfully!"

conda deactivate

echo "Results saved under $OUTPUT_DIR"
