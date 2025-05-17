#!/bin/bash
# groundtruth.sh - Script to process and copy ground truth files with flexible format support
set -euo pipefail # Enable strict error handling

# Function to display usage information
usage() {
    cat << EOF
Usage: $0 [OPTIONS]
Required options:
    -training_path <path>    Path to training data directory
    -output_path <path>      Path for output file
Optional options:
    -format <ext>           Specify ground truth file format (default: attempts to find csv/json/jsonl/txt/pq)
    -h, --help             Display this help message
EOF
    exit 1
}

# Function to log messages with timestamp
log_message() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Function to find ground truth file
find_ground_truth() {
    local base_path="$1"
    local format="${2:-}"
    
    if [ -n "$format" ]; then
        local gt_path="${base_path}/eval/ground_truth.${format}"
        if [ -f "$gt_path" ]; then
            echo "$gt_path"
            return 0
        fi
        log_message "Error: Ground truth file with format ${format} not found"
        return 1
    else
        # Try common formats
        for ext in csv json jsonl txt pq; do
            local gt_path="${base_path}/eval/ground_truth.${ext}"
            if [ -f "$gt_path" ]; then
                echo "$gt_path"
                return 0
            fi
        done
        log_message "Error: No ground truth file found in supported formats"
        return 1
    fi
}

# Parse command line arguments
TRAINING_PATH=""
OUTPUT_PATH=""
FORMAT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -training_path)
            TRAINING_PATH="$2"
            shift 2
            ;;
        -output_path)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        -format)
            FORMAT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            log_message "Error: Unknown option $1"
            usage
            ;;
    esac
done

# Validate required arguments
if [ -z "$TRAINING_PATH" ] || [ -z "$OUTPUT_PATH" ]; then
    log_message "Error: Missing required arguments"
    usage
fi

# Validate training path exists
if [ ! -d "$TRAINING_PATH" ]; then
    log_message "Error: Training path directory does not exist: $TRAINING_PATH"
    exit 1
fi

# Create output directory if it doesn't exist
OUTPUT_DIR=$(dirname "$OUTPUT_PATH")
mkdir -p "$OUTPUT_DIR"

# Extract the dataset name from the training path
DATASET_NAME=$(basename "$(dirname "$TRAINING_PATH")")

# Construct the path to the ground truth file
GROUND_TRUTH_BASE="/media/agent/maab/datasets/${DATASET_NAME}"

# Find ground truth file
GROUND_TRUTH_PATH=$(find_ground_truth "$GROUND_TRUTH_BASE" "$FORMAT") || exit 1

# Create a results directory
RESULTS_DIR="${OUTPUT_DIR}"
mkdir -p "$RESULTS_DIR"

# Copy the ground truth file to the output path
if cp "$GROUND_TRUTH_PATH" "$OUTPUT_PATH"; then
    log_message "Successfully copied ground truth file to $OUTPUT_PATH"
    log_message "Source format: $(basename "$GROUND_TRUTH_PATH" | sed 's/.*\.//')"
else
    log_message "Error: Failed to copy ground truth file"
    exit 1
fi

# Copy everything under eval directory to results
EVAL_DIR="${GROUND_TRUTH_BASE}/eval"
if [ -d "$EVAL_DIR" ]; then
    if cp -r "$EVAL_DIR"/* "$RESULTS_DIR/"; then
        log_message "Successfully copied eval directory contents to $RESULTS_DIR"
    else
        log_message "Error: Failed to copy eval directory contents"
        exit 1
    fi
else
    log_message "Warning: Eval directory not found at $EVAL_DIR"
fi