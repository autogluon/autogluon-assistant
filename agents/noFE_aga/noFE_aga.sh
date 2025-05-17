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

eval "$(conda shell.bash hook)"
conda activate aga || { echo "Failed to activate conda environment 'aga'"; exit 1; }
aga run "${TRAINING_PATH}" --presets medium_quality --output-filename "${OUTPUT_DIR}/results.csv" \
        2>&1 | tee "${OUTPUT_DIR}/log.txt"
conda deactivate

echo "Results saved under $OUTPUT_DIR"
