# MAAB (Multimodal AutoML Agent Benchmark)

MAAB is a comprehensive benchmark framework designed to evaluate AutoML agents across diverse multimodal datasets. This benchmark helps researchers and practitioners assess the performance, efficiency, and robustness of AutoML solutions in handling various types of data.

## Features

- Standardized evaluation framework for AutoML agents
- Support for multiple data modalities (text, image, tabular, etc.)
- Automated performance metrics collection
- Extensive dataset collection covering various domains
- Scalable architecture for batch evaluation

## Installation

1. Enter the repository:
```bash
cd maab
```

2. Download and extract the benchmark datasets (an 140GiB archive file, 159GiB after unzip):
```bash
wget https://mlzero.s3.us-east-1.amazonaws.com/data/maabdatasets20250504.zip
unzip maabdatasets20250504.zip
```

3. Set up the Python environment:
```bash
conda create -n maab -y
conda activate maab
pip install -r requirements.txt
```

4. (To benchmark MLZero) Install the AutoMLAgent package (See AutoMLAgent repo for more details):
```bash
cd ../AutoMLAgent
conda deactivate
conda create -n agent -y
conda activate agent
pip install -e .
conda deactivate
```

## Usage

### Basic Evaluation

To evaluate one or more agents on specific datasets:

```bash
eval_batch.sh -a <agent_name1,agent_name2,...|all> -d <dataset_name1,dataset_name2,...|all>
```

Parameters:
- `-a`: Specify agent names (comma-separated) or use 'all' for evaluating all available agents
- `-d`: Specify dataset names (comma-separated) or use 'all' for evaluating on all datasets

Example:
```bash
eval_batch.sh -a agentsonnet37 -d abalone
```

### Default Agent for MLZero

- agentsonnet37

## Datasets

MAAB includes a diverse collection of datasets covering various domains and problem types. The datasets are organized in a standardized structure under `maab/datasets/`:

```
maab/datasets/
└── [dataset_name]/
    ├── training/      # Only this folder is exposed to agents
    ├── eval/          # Hidden evaluation data
    ├── backups/       # Backup files
    └── metadata.json  # Dataset configuration and metadata
```

### Available Datasets

Note: Agents only have access to the `training` folder during development and evaluation. The `eval` folder contains hidden test data used for final performance assessment.

Each dataset includes a `metadata.json` file that specifies:
- Problem type
- Evaluation metric
- ...

## Output Structure

The evaluation results are organized in timestamped run directories under `maab/runs/`. Each run has the following structure:

```
maab/runs/RUN_[TIMESTAMP]/
├── outputs/
│   └── [agent_name]_[dataset_name]_output/
│       └── ... (agent-specific output files)
└── overall_results.csv
```

- Each run is stored in a directory named with the pattern `RUN_[TIMESTAMP]` (e.g., `RUN_20250212_235513`)
- The `outputs` directory contains subdirectories for each agent-dataset combination
- Each agent-dataset output directory is named as `[agent_name]_[dataset_name]_output`
- `overall_results.csv` contains the consolidated evaluation metrics for all agent-dataset combinations in the run

## Metrics

MAAB supports a comprehensive set of evaluation metrics based on the problem type:

### Classification (Binary & Multiclass)
- Accuracy: Standard classification accuracy
- F1 (Weighted): Weighted average of F1 scores
- F1: Standard F1 score
- AUROC: Area Under the Receiver Operating Characteristic curve
- NLL: Negative Log-Likelihood

### Regression
- RMSE: Root Mean Square Error
- RMSLE: Root Mean Square Logarithmic Error
- MAE: Mean Absolute Error
- MEDAE: Median Absolute Error
- R-Squared: Coefficient of determination

### Semantic Segmentation
- IoU: Intersection over Union
- S-Measure: Structure-measure evaluation

### Time Series
- SQL: Supervised Quality Loss
- WQL: Weighted Quality Loss
- MAE: Mean Absolute Error
- MASE: Mean Absolute Scaled Error
- WAPE: Weighted Absolute Percentage Error
- MSE: Mean Squared Error
- RMSE: Root Mean Square Error
- RMSLE: Root Mean Square Logarithmic Error
- RMSSE: Root Mean Square Scaled Error
- MAPE: Mean Absolute Percentage Error
- SMAPE: Symmetric Mean Absolute Percentage Error

To specify a metric for evaluation, use the `metric_name` field in the dataset metadata. The metric name should be lowercase and match one of the supported metrics for the given problem type.

## License

MAAB is released under the Apache 2.0 License. See the LICENSE file for more details.
