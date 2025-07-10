# Condensed: 

Summary: This tutorial provides implementation guidance for setting up the ML50 dataset for machine translation tasks. It covers installation requirements, data downloading procedures, and preprocessing steps using SentencePiece tokenization. The tutorial explains environment variable configuration and outlines the dataset's directory structure with raw, deduplicated, and cleaned data folders. This knowledge helps with preparing multilingual translation datasets, implementing data preprocessing pipelines, and organizing machine translation training data effectively. Key functionalities include dependency installation, data organization, deduplication, and test set separation.

*This is a condensed version that preserves essential implementation details and context.*

# ML50 Dataset Setup

## Installation
```bash
pip install -r requirement.txt
```

## Data Download
```bash
export WORKDIR_ROOT=<a directory which will hold all working files>
```
The downloaded data will be stored in `$WORKDIR_ROOT/ML50`

## Data Preprocessing
1. Install SentencePiece from [GitHub](https://github.com/google/sentencepiece)
2. Set environment variables:
   ```bash
   export WORKDIR_ROOT=<a directory which will hold all working files>
   export SPM_PATH=<a path pointing to sentencepiece spm_encode.py>
   ```

## Directory Structure
- `$WORKDIR_ROOT/ML50/raw`: Extracted raw data
- `$WORKDIR_ROOT/ML50/dedup`: Deduplicated data
- `$WORKDIR_ROOT/ML50/clean`: Data with valid and test sentences removed from the deduplicated data