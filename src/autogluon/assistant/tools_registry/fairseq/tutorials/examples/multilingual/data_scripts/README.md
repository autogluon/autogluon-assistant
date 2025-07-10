Summary: This tutorial provides implementation guidance for setting up the ML50 dataset for machine translation tasks. It covers installation requirements, data downloading procedures, and preprocessing steps using SentencePiece tokenization. The tutorial explains environment variable configuration and outlines the dataset's directory structure with raw, deduplicated, and cleaned data folders. This knowledge helps with preparing multilingual translation datasets, implementing data preprocessing pipelines, and organizing machine translation training data effectively. Key functionalities include dependency installation, data organization, deduplication, and test set separation.


# Install dependency
```bash
pip install -r requirement.txt
```

# Download the data set
```bash
export WORKDIR_ROOT=<a directory which will hold all working files>

```
The downloaded data will be at $WORKDIR_ROOT/ML50

# preprocess the data
Install SPM [here](https://github.com/google/sentencepiece)
```bash
export WORKDIR_ROOT=<a directory which will hold all working files>
export SPM_PATH=<a path pointing to sentencepice spm_encode.py>
```
* $WORKDIR_ROOT/ML50/raw: extracted raw data
* $WORKDIR_ROOT/ML50/dedup: dedup data
* $WORKDIR_ROOT/ML50/clean: data with valid and test sentences removed from the dedup data
 

