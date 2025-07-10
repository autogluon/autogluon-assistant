Summary: This tutorial implements CRISS (Cross-lingual Retrieval for Iterative Self-Supervised Training), a multilingual sequence-to-sequence pretraining method. It covers: (1) unsupervised machine translation with pre-trained checkpoints, (2) cross-lingual sentence retrieval using the Tatoeba dataset, and (3) mining pseudo-parallel data with FAISS. Key functionalities include evaluating translation between language pairs (e.g., Sinhala-English), performing sentence retrieval across languages (e.g., Kazakh-English), and implementing the iterative mining process that improves cross-lingual alignment. The code demonstrates practical applications of cross-lingual embedding spaces for NLP tasks.

# Cross-lingual Retrieval for Iterative Self-Supervised Training

https://arxiv.org/pdf/2006.09526.pdf

## Introduction

CRISS is a multilingual sequence-to-sequnce pretraining method where mining and training processes are applied iteratively, improving cross-lingual alignment and translation ability at the same time.

## Requirements:

* faiss: https://github.com/facebookresearch/faiss
* mosesdecoder: https://github.com/moses-smt/mosesdecoder
* flores: https://github.com/facebookresearch/flores
* LASER: https://github.com/facebookresearch/LASER

## Unsupervised Machine Translation
##### 1. Download and decompress CRISS checkpoints
```
cd examples/criss
wget https://dl.fbaipublicfiles.com/criss/criss_3rd_checkpoints.tar.gz
tar -xf criss_checkpoints.tar.gz
```
##### 2. Download and preprocess Flores test dataset
Make sure to run all scripts from examples/criss directory
```
bash download_and_preprocess_flores_test.sh
```

##### 3. Run Evaluation on Sinhala-English
```
bash unsupervised_mt/eval.sh
```

## Sentence Retrieval
##### 1. Download and preprocess Tatoeba dataset
```
bash download_and_preprocess_tatoeba.sh
```

##### 2. Run Sentence Retrieval on Tatoeba Kazakh-English
```
bash sentence_retrieval/sentence_retrieval_tatoeba.sh
```

## Mining
##### 1. Install faiss
Follow instructions on https://github.com/facebookresearch/faiss/blob/master/INSTALL.md
##### 2. Mine pseudo-parallel data between Kazakh and English
```
bash mining/mine_example.sh
```

## Citation
```bibtex
@article{tran2020cross,
  title={Cross-lingual retrieval for iterative self-supervised training},
  author={Tran, Chau and Tang, Yuqing and Li, Xian and Gu, Jiatao},
  journal={arXiv preprint arXiv:2006.09526},
  year={2020}
}
```
