#!/bin/bash
# MAAB Evaluation Script for Multiple Agents and Datasets

# Function to display usage information
usage() {
    echo "Usage: $0 -a <agent_name1,agent_name2,...|all> -d <dataset_name1,dataset_name2,...|all|ablation>"
    echo " -a: Names of agents to evaluate (comma-separated) or 'all'"
    echo " -d: Names of datasets to use (comma-separated) or 'all' or 'ablation'"
    echo "      Note: 'ablation' is a special preset that includes a specific set of datasets"
    exit 1
}

# Function to get list of all agents
get_all_agents() {
    local agents_dir="${MAAB_DIR}/agents"
    ls -1 "$agents_dir" 2>/dev/null
}

# Function to get list of all datasets
get_all_datasets() {
    local datasets_dir="${MAAB_DIR}/datasets"
    ls -1 "$datasets_dir" 2>/dev/null
}

# Function to get the ablation dataset list
get_ablation_datasets() {
    echo "yolanda,mldoc,europeanflooddepth,petfinder,camo_sem_seg,rvl_cdip,solar_10_minutes,fiqabeir"
}

# Parse command-line arguments
while getopts "a:d:" opt; do
    case $opt in
        a) AGENTS="$OPTARG" ;;
        d) DATASETS="$OPTARG" ;;
        *) usage ;;
    esac
done

# Check if both agents and datasets are provided
if [ -z "$AGENTS" ] || [ -z "$DATASETS" ]; then
    usage
fi

# Set the base directory for MAAB
MAAB_DIR="$(dirname "$(realpath "$0")")"

# Generate a timestamp for the evaluation run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${MAAB_DIR}/runs/RUN_${TIMESTAMP}"
OUTPUT_DIR="${RUN_DIR}/outputs"
RESULTS_FILE="${RUN_DIR}/overall_results.csv"
TIMING_FILE="${RUN_DIR}/execution_times.csv"

# Function to run an agent on a dataset
run_agent() {
    local agent=$1
    local dataset=$2
    local training_path="${MAAB_DIR}/datasets/${dataset}/training"
    local agent_output_dir="${OUTPUT_DIR}/${agent}_${dataset}_output"
    local eval_dir="${MAAB_DIR}/datasets/${dataset}/eval"
    
    mkdir -p "$agent_output_dir"
    
    echo "Running ${agent} on ${dataset}..."
    
    # Record start time
    local start_time=$(date +%s)
    
    # Run the agent script
    bash "${MAAB_DIR}/agents/${agent}/${agent}.sh" \
        -training_path "$training_path" \
        -output_dir "$agent_output_dir"

    # Record end time and calculate duration
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # Log the timing information
    echo "${agent},${dataset},${duration}" >> "$TIMING_FILE"
    
    # Check if agent script execution was successful
    if [ $? -ne 0 ]; then
        echo "Error: Agent script failed for ${agent} on ${dataset}"
        return 1
    fi
    
    # Find the results file
    local pred_path=$(ls ${agent_output_dir}/results.* 2>/dev/null | head -n 1)
    if [ -z "$pred_path" ]; then
        echo "Error: Could not find results file for ${agent} on ${dataset}"
        return 1
    fi
    
    # Find the ground truth file
    local gt_path=$(ls ${eval_dir}/ground_truth.* 2>/dev/null | head -n 1)
    if [ -z "$gt_path" ]; then
        echo "Error: Could not find ground truth file for dataset ${dataset}"
        return 1
    fi
    
    # Run evaluation
    python3 "${MAAB_DIR}/tools/evaluators.py" \
        --pred_path "$pred_path" \
        --gt_path "$gt_path" \
        --metadata_path "${MAAB_DIR}/datasets/${dataset}/metadata.json" \
        --results_path "$RESULTS_FILE" \
        --agent_name "$agent" \
        2>&1 | tee "${agent_output_dir}/eval_log.txt"
}

# Function to validate agent exists
validate_agent() {
    local agent=$1
    if [ ! -d "${MAAB_DIR}/agents/${agent}" ]; then
        echo "Error: Agent '${agent}' not found in ${MAAB_DIR}/agents/"
        return 1
    fi
    return 0
}

# Function to validate dataset exists
validate_dataset() {
    local dataset=$1
    if [ ! -d "${MAAB_DIR}/datasets/${dataset}" ]; then
        echo "Error: Dataset '${dataset}' not found in ${MAAB_DIR}/datasets/"
        return 1
    fi
    return 0
}

# Main execution
main() {
    # Create run directory and output directory
    mkdir -p "$OUTPUT_DIR"
    
    # Process agents list
    if [ "$AGENTS" = "all" ]; then
        AGENT_LIST=($(get_all_agents))
    else
        IFS=',' read -ra AGENT_LIST <<< "$AGENTS"
    fi
    
    # Process datasets list
    if [ "$DATASETS" = "all" ]; then
        DATASET_LIST=($(get_all_datasets))
    elif [ "$DATASETS" = "ablation" ]; then
        # Use the predefined ablation dataset list
        IFS=',' read -ra DATASET_LIST <<< "$(get_ablation_datasets)"
        echo "Using ablation dataset set: ${DATASETS}"
    else
        IFS=',' read -ra DATASET_LIST <<< "$DATASETS"
    fi
    
    # Validate all agents and datasets before starting
    for agent in "${AGENT_LIST[@]}"; do
        if ! validate_agent "$agent"; then
            exit 1
        fi
    done
    
    for dataset in "${DATASET_LIST[@]}"; do
        if ! validate_dataset "$dataset"; then
            exit 1
        fi
    done
    
    # Create header for timing file
    echo "agent,dataset,execution_time_seconds" > "$TIMING_FILE"
    
    # Run each agent on each dataset
    for agent in "${AGENT_LIST[@]}"; do
        for dataset in "${DATASET_LIST[@]}"; do
            if ! run_agent "$agent" "$dataset"; then
                echo "Warning: Failed to process ${agent} on ${dataset}"
                continue
            fi
        done
    done
    
    # Generate a human-readable timing report
    {
        echo "Execution Time Report for Run: ${TIMESTAMP}"
        echo "========================================"
        echo ""
        echo "Agent-Dataset Pairs (sorted by execution time):"
        echo ""
        
        # Sort the timing file by execution time (descending) and format for readability
        tail -n +2 "$TIMING_FILE" | sort -t, -k3,3nr | while IFS=, read -r agent dataset seconds; do
            minutes=$((seconds / 60))
            remaining_seconds=$((seconds % 60))
            echo "  ${agent} on ${dataset}: ${minutes}m ${remaining_seconds}s (${seconds} seconds total)"
        done
        
        echo ""
        echo "Summary by Agent:"
        echo ""
        
        # Generate summary statistics per agent
        tail -n +2 "$TIMING_FILE" | awk -F, '
        {
            agents[$1] = 1
            agent_times[$1] += $3
            if (!agent_count[$1]) agent_count[$1] = 0
            agent_count[$1]++
            if (!agent_max[$1] || $3 > agent_max[$1]) agent_max[$1] = $3
            if (!agent_min[$1] || $3 < agent_min[$1]) agent_min[$1] = $3
        }
        END {
            for (agent in agents) {
                avg = agent_times[agent] / agent_count[agent]
                printf "  %s: Avg=%ds, Min=%ds, Max=%ds, Total=%ds\n", 
                       agent, avg, agent_min[agent], agent_max[agent], agent_times[agent]
            }
        }'
        
    } > "${RUN_DIR}/timing_report.txt"
    
    echo "Evaluation complete. Results are stored in ${RESULTS_FILE}"
    echo "Execution times are stored in ${TIMING_FILE}"
    echo "A detailed timing report is available at ${RUN_DIR}/timing_report.txt"
}

# Run the main function
main
