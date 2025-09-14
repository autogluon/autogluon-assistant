#!/bin/bash
set -e

# Set variables
RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_QUEUE="MLEBench-Job-Queue"
JOB_DEFINITION="MLE-Bench-Job-Definition"
REGION="us-east-1"  # Set your desired AWS region here

# Display usage information
usage() {
    echo "Usage: $0 -a <agent_name1,agent_name2,...|all> -d <dataset_name1,dataset_name2,...|all|ablation>"
    echo " -a: Names of agents to evaluate (comma-separated) or 'all'"
    echo " -d: Names of datasets to use (comma-separated) or 'all' or 'ablation'"
    echo "      Note: 'ablation' is a special preset that includes a specific set of datasets"
    exit 1
}

# Function to get list of all agents
get_all_agents() {
    ls -1 "/fsx/mlzero-dev/autogluon-assistant/maab/agents"
}

# Function to get list of all datasets
get_all_datasets() {
    ls -1 "/fsx/mlzero-dev/autogluon-assistant/maab/datasets"
}

# Function to get the ablation dataset list
get_ablation_datasets() {
    echo "yolanda,mldoc,europeanflooddepth,petfinder,camo_sem_seg,rvl_cdip,solar_10_minutes,fiqabeir"
}

# Parse command line arguments
while getopts "a:d:r:" opt; do
    case $opt in
        a) AGENTS="$OPTARG" ;;
        d) DATASETS="$OPTARG" ;;
        r) AWS_REGION="$OPTARG" ;;
        *) usage ;;
    esac
done

# Check if both agents and datasets are provided
if [ -z "$AGENTS" ] || [ -z "$DATASETS" ]; then
    usage
fi

# Process agents list
if [ "$AGENTS" = "all" ]; then
    IFS=$'\n' read -d '' -ra AGENT_LIST < <(get_all_agents)
else
    IFS=',' read -ra AGENT_LIST <<< "$AGENTS"
fi

# Process datasets list
if [ "$DATASETS" = "all" ]; then
    IFS=$'\n' read -d '' -ra DATASET_LIST < <(get_all_datasets)
elif [ "$DATASETS" = "ablation" ]; then
    IFS=',' read -ra DATASET_LIST <<< "$(get_ablation_datasets)"
    echo "Using ablation dataset set: ${DATASETS}"
else
    IFS=',' read -ra DATASET_LIST <<< "$DATASETS"
fi

# Create run directory
mkdir -p "/fsx/mlzero-dev/autogluon-assistant/maab/runs/RUN_${RUN_TIMESTAMP}/outputs"
RESULTS_FILE="/fsx/mlzero-dev/autogluon-assistant/maab/runs/RUN_${RUN_TIMESTAMP}/overall_results.csv"

# Create header for results file
echo "agent,metric,value" > "$RESULTS_FILE"

# Validate agents and datasets
echo "Validating agents and datasets..."
for agent in "${AGENT_LIST[@]}"; do
    if [ ! -d "/fsx/mlzero-dev/autogluon-assistant/maab/agents/${agent}" ]; then
        echo "Error: Agent '${agent}' not found in /fsx/mlzero-dev/autogluon-assistant/maab/agents/"
        exit 1
    fi
done

for dataset in "${DATASET_LIST[@]}"; do
    if [ ! -d "/fsx/mlzero-dev/autogluon-assistant/maab/datasets/${dataset}" ]; then
        echo "Error: Dataset '${dataset}' not found in /fsx/mlzero-dev/autogluon-assistant/maab/datasets/"
        exit 1
    fi
done

# Create job tracking file
JOB_TRACKING_FILE="/fsx/mlzero-dev/autogluon-assistant/maab/runs/RUN_${RUN_TIMESTAMP}/job_tracking.csv"
echo "job_id,agent,dataset,status,submission_time" > "$JOB_TRACKING_FILE"

# Submit jobs
echo "Submitting batch jobs for run: RUN_${RUN_TIMESTAMP}"
echo "Total jobs to submit: $((${#AGENT_LIST[@]} * ${#DATASET_LIST[@]}))"

for agent in "${AGENT_LIST[@]}"; do
    for dataset in "${DATASET_LIST[@]}"; do
        JOB_NAME="${agent}_${dataset}_${RUN_TIMESTAMP}"
        echo "Submitting job: $JOB_NAME"

        # Submit the job
        aws batch submit-job \
          --region "$REGION" \
          --job-name "$JOB_NAME" \
          --job-queue "$JOB_QUEUE" \
          --job-definition "$JOB_DEFINITION"

        if [ $? -eq 0 ]; then
            echo "Job submitted: $JOB_NAME"
            echo "$job_id,$agent,$dataset,SUBMITTED,$(date +%Y-%m-%d-%H:%M:%S)" >> "$JOB_TRACKING_FILE"
        else
            echo "Failed to submit job: $JOB_NAME"
            echo "FAILED,$agent,$dataset,SUBMISSION_FAILED,$(date +%Y-%m-%d-%H:%M:%S)" >> "$JOB_TRACKING_FILE"
        fi
        
        # Small delay to avoid API rate limiting
        sleep 1
    done
done

echo "All jobs submitted for run: RUN_${RUN_TIMESTAMP}"
echo "Results will be available in: /fsx/mlzero-dev/autogluon-assistant/maab/runs/RUN_${RUN_TIMESTAMP}"
echo "To monitor job status, run: /fsx/mlzero-dev/autogluon-assistant/maab/runs/RUN_${RUN_TIMESTAMP}/monitor_jobs.sh"