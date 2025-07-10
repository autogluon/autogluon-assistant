# Condensed: HuBERT

Summary: This tutorial provides implementation guidance for HuBERT speech models, covering pre-trained model options (Base, Large, Extra Large) with download links and parameter counts. It demonstrates code for loading models, and outlines a complete training pipeline including data preparation, pre-training, and fine-tuning with CTC loss. The tutorial details three decoding methods (Viterbi/greedy, KenLM n-gram, and Fairseq-LM) with sample code and configurable parameters. This resource helps with speech recognition tasks, audio feature extraction, and implementing self-supervised speech models with various decoding strategies.

*This is a condensed version that preserves essential implementation details and context.*

# HuBERT Implementation Guide

## Pre-trained Models Overview

| Model Type | Parameters | Download Links |
|---|---|---|
| HuBERT Base | ~95M | [Pretrained Model](https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt), [L9 Quantizer](https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960_L9_km500.bin) |
| HuBERT Large | ~316M | [Pretrained](https://dl.fbaipublicfiles.com/hubert/hubert_large_ll60k.pt), [Fine-tuned](https://dl.fbaipublicfiles.com/hubert/hubert_large_ll60k_finetune_ls960.pt) |
| HuBERT Extra Large | ~1B | [Pretrained](https://dl.fbaipublicfiles.com/hubert/hubert_xtralarge_ll60k.pt), [Fine-tuned](https://dl.fbaipublicfiles.com/hubert/hubert_xtralarge_ll60k_finetune_ls960.pt) |

## Loading a Model

```python
ckpt_path = "/path/to/the/checkpoint.pt"
models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
model = models[0]
```

## Training Pipeline

### 1. Data Preparation

Follow steps in `./simple_kmeans` to create:
- `{train,valid}.tsv` - waveform list files
- `{train,valid}.km` - frame-aligned pseudo label files
- `dict.km.txt` - dictionary file

**Important**: The `label_rate` must match the feature frame rate used for clustering (100Hz for MFCC, 50Hz for HuBERT features).

### 2. Pre-training

```sh
python fairseq_cli/hydra_train.py \
  --config-dir /path/to/fairseq-py/examples/hubert/config/pretrain \
  --config-name hubert_base_librispeech \
  task.data=/path/to/data task.label_dir=/path/to/labels task.labels='["km"]' model.label_rate=100
```

### 3. Fine-tuning with CTC Loss

```sh
python fairseq_cli/hydra_train.py \
  --config-dir /path/to/fairseq-py/examples/hubert/config/finetune \
  --config-name base_10h \
  task.data=/path/to/data task.label_dir=/path/to/trans \
  model.w2v_path=/path/to/checkpoint
```

## Decoding Methods

### 1. Viterbi Decoding (Greedy)

```sh
python examples/speech_recognition/new/infer.py \
  --config-dir /path/to/fairseq-py/examples/hubert/config/decode \
  --config-name infer_viterbi \
  task.data=/path/to/data \
  task.normalize=[true|false] \
  decoding.exp_dir=/path/to/experiment/directory \
  common_eval.path=/path/to/checkpoint \
  dataset.gen_subset=test
```

### 2. KenLM Decoding (with n-gram LM)

```sh
python examples/speech_recognition/new/infer.py \
  --config-dir /path/to/fairseq-py/examples/hubert/config/decode \
  --config-name infer_kenlm \
  task.data=/path/to/data \
  task.normalize=[true|false] \
  decoding.exp_dir=/path/to/experiment/directory \
  common_eval.path=/path/to/checkpoint \
  dataset.gen_subset=test \
  decoding.decoder.lexicon=/path/to/lexicon \
  decoding.decoder.lmpath=/path/to/arpa
```

### 3. Fairseq-LM Decoding

Use `--config-name infer_fsqlm` instead of `infer_kenlm` with appropriate lexicon and LM paths.

## Critical Decoding Parameters

Key parameters that can be configured:
- `decoding.decoder.beam` - beam size (e.g., 500)
- `decoding.decoder.beamthreshold` - pruning threshold
- `decoding.decoder.lmweight` - language model weight
- `decoding.decoder.wordscore` - word insertion score
- `decoding.decoder.silweight` - silence weight