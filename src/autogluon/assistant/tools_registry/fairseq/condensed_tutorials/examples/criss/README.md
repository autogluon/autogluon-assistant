# Condensed: Cross-lingual Retrieval for Iterative Self-Supervised Training

Summary: This tutorial implements CRISS (Cross-lingual Retrieval for Iterative Self-Supervised Training), a multilingual sequence-to-sequence pretraining method. It covers: (1) unsupervised machine translation with pre-trained checkpoints, (2) cross-lingual sentence retrieval using the Tatoeba dataset, and (3) mining pseudo-parallel data with FAISS. Key functionalities include evaluating translation between language pairs (e.g., Sinhala-English), performing sentence retrieval across languages (e.g., Kazakh-English), and implementing the iterative mining process that improves cross-lingual alignment. The code demonstrates practical applications of cross-lingual embedding spaces for NLP tasks.

*This is a condensed version that preserves essential implementation details and context.*

# Cross-lingual Retrieval for Iterative Self-Supervised Training (CRISS)

## Overview
CRISS is a multilingual sequence-to-sequence pretraining method that iteratively applies mining and training processes to improve cross-lingual alignment and translation capabilities simultaneously.

## Requirements
- faiss: https://github.com/facebookresearch/faiss
- mosesdecoder: https://github.com/moses-smt/mosesdecoder
- flores: https://github.com/facebookresearch/flores
- LASER: https://github.com/facebookresearch/LASER

## Implementation Steps

### Unsupervised Machine Translation
1. **Download and setup CRISS checkpoints**
   ```bash
   cd examples/criss
   wget https://dl.fbaipublicfiles.com/criss/criss_3rd_checkpoints.tar.gz
   tar -xf criss_checkpoints.tar.gz
   ```

2. **Prepare Flores test dataset**
   ```bash
   bash download_and_preprocess_flores_test.sh
   ```

3. **Run evaluation (Sinhala-English example)**
   ```bash
   bash unsupervised_mt/eval.sh
   ```

### Sentence Retrieval
1. **Prepare Tatoeba dataset**
   ```bash
   bash download_and_preprocess_tatoeba.sh
   ```

2. **Run retrieval (Kazakh-English example)**
   ```bash
   bash sentence_retrieval/sentence_retrieval_tatoeba.sh
   ```

### Mining Parallel Data
1. **Install faiss** following instructions at https://github.com/facebookresearch/faiss/blob/master/INSTALL.md

2. **Mine pseudo-parallel data**
   ```bash
   bash mining/mine_example.sh
   ```

## Important Notes
- All scripts should be run from the `examples/criss` directory
- The implementation demonstrates mining and evaluation on specific language pairs (Sinhala-English, Kazakh-English)