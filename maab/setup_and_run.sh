#!/bin/bash
# Setup and Run Script for MAAB AWS Batch Jobs
# This script is run as the entrypoint for the Docker container

set -e  # Exit on error

# Define log function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log "Starting MAAB evaluation container"
log "Agent: $AGENT_NAME"
log "Dataset: $DATASET_NAME"
log "Run timestamp: $RUN_TIMESTAMP"

# Mount FSx file system
#amazon-linux-extras install -y lustre2.10
#amazon-linux-extras install -y lustre2.10
#wget -O - https://fsx-lustre-client-repo-public-keys.s3.amazonaws.com/fsx-ubuntu-public-key.asc | gpg --dearmor | tee /usr/share/keyrings/fsx-ubuntu-public-key.gpg >/dev/null
#bash -c 'echo "deb [signed-by=/usr/share/keyrings/fsx-ubuntu-public-key.gpg] https://fsx-lustre-client-repo.s3.amazonaws.com/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/fsxlustreclientrepo.list && apt-get update'
#apt install -y lustre-client-modules-$(uname -r)

# DEBUG FSX: Quick connectivity test
#nc -v fs-061a432604102fdc4.fsx.us-east-1.amazonaws.com 988
# Check if Lustre is ready
#lsmod | grep lustre
# View detailed logs
#grep -E "(✅|❌)" /var/log/fsx-debug.log

#mkdir -p /fsx
ls /fsx/mlzero-dev
#mount -t lustre -o relatime,flock fs-061a432604102fdc4.fsx.us-east-1.amazonaws.com@tcp:/rfaatbuv /fsx/mlzero-dev
#if [ $? -ne 0 ]; then
#    log "ERROR: Failed to mount FSx file system"
#    exit 1
#fi
log "FSx mounted successfully at /fsx/mlzero-dev"

# Setup logging to both console and file
exec > >(tee -a "/fsx/mlzero-dev/autogluon-assistant/maab/container_log.txt")
exec 2>&1

RUN_DIR="/fsx/mlzero-dev/autogluon-assistant/maab/runs/RUN_${RUN_TIMESTAMP}"
OUTPUT_DIR="${RUN_DIR}/outputs"
mkdir -p "$OUTPUT_DIR"

# Setup MLZero environment
log "Setting up MLZero environment"
source /opt/conda/etc/profile.d/conda.sh
conda activate mlzero
if [ $? -ne 0 ]; then
    log "ERROR: Failed to activate mlzero conda environment"
    exit 1
fi

cd /fsx/mlzero-dev/autogluon-assistant
log "Installing MLZero from $(pwd)"
pip install uv
uv pip install opencv-python-headless
uv pip install -e . || log "WARNING: Error during MLZero installation"
conda deactivate

# Setup MAAB environment
log "Setting up MAAB environment"
conda activate maab
if [ $? -ne 0 ]; then
    log "ERROR: Failed to activate maab conda environment"
    exit 1
fi

cd /fsx/mlzero-dev/autogluon-assistant/maab
log "Installing MAAB from $(pwd)"
pip install uv
uv pip install opencv-python-headless
uv pip install -r requirements.txt || log "WARNING: Error installing MAAB requirements"

# Execute the agent-dataset evaluation using the AWS Batch specific script
log "Starting evaluation"
cd /fsx/mlzero-dev/autogluon-assistant/maab

# Use the dedicated AWS Batch evaluation script
log "Running eval_aws_batch.sh for ${AGENT_NAME} on ${DATASET_NAME}"
bash "/fsx/mlzero-dev/autogluon-assistant/maab/eval_aws_batch.sh" \
    -a "$AGENT_NAME" \
    -d "$DATASET_NAME" \
    -t "$RUN_TIMESTAMP" \
    2>&1 | tee -a "${OUTPUT_DIR}/${AGENT_NAME}_${DATASET_NAME}_output/container_log.txt"

log "Container execution complete"