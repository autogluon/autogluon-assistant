# Condensed: MR-HuBERT

Summary: This tutorial provides implementation guidance for MR-HuBERT (Multi-Resolution HuBERT), a speech representation model. It covers loading pre-trained models (base, large, and multilingual variants) using fairseq, training new models through data preparation (creating TSV files, pseudo labels, and dictionaries), pre-training with multi-resolution parameters, and fine-tuning with CTC loss. The tutorial details three decoding approaches: Viterbi (greedy) decoding, KenLM decoding with n-gram language models, and Fairseq-LM decoding with neural language models. Key parameters for training (label_rate, label_rate_ratios) and decoding (beam size, thresholds, weights) are highlighted with code examples for each implementation step.

*This is a condensed version that preserves essential implementation details and context.*

# MR-HuBERT Implementation Guide

## Pre-trained Models

MR-HuBERT offers several pre-trained models in different sizes:
- **Base models (~97M parameters)**: Trained on Librispeech 960hr
- **Large models (~321M parameters)**: Trained on Libri-Light 60k hr
- **Multilingual models**: Trained on Voxpopuli 100k hr

Various ablation models are also available with different layer configurations and training approaches.

## Using Pre-trained Models

```python
# Loading a pre-trained model
ckpt_path = "/path/to/the/checkpoint.pt"
models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
model = models[0]
```

## Training New Models

### Data Preparation

1. Create TSV files with waveform paths and lengths:
   ```
   /path/to/audio/file1.wav\t160000
   /path/to/audio/file2.wav\t154600
   ```

2. Create frame-aligned pseudo label files (`.km`):
   ```
   44 44 44 48 48 962 962 962 962 962 962 962 962 967 967 967...
   ```

3. Create a dictionary file (`dict.km.txt`):
   ```
   0 1
   1 1
   ...
   999 1
   ```

### Pre-training

```bash
python fairseq_cli/hydra_train.py \
  --config-dir /path/to/fairseq-py/examples/mr_hubert/config/pretrain \
  --config-name mrhubert_base_librispeech \
  task.data=/path/to/data task.label_dir=/path/to/labels \
  task.labels='["km"]' model.label_rate=100 \
  task.label_rate_ratios='[1, 2]'
```

**Important parameters:**
- `task.label_rate`: Feature frame rate used for clustering (100Hz for MFCC, 50Hz for HuBERT)
- `task.label_rate_ratios`: Ratios for multi-resolution training

### Fine-tuning with CTC Loss

1. Prepare character transcripts (`.ltr` files):
   ```
   HOW | ARE | YOU
   ...
   THANK | YOU
   ```

2. Run fine-tuning:
   ```bash
   python fairseq_cli/hydra_train.py \
     --config-dir /path/to/fairseq-py/examples/mr_hubert/config/finetune \
     --config-name base_10h \
     task.data=/path/to/data task.label_dir=/path/to/trans \
     model.w2v_path=/path/to/checkpoint
   ```

## Decoding

MR-HuBERT supports three decoding modes:

### 1. Viterbi Decoding (Greedy)

```bash
python examples/speech_recognition/new/infer.py \
  --config-dir /path/to/fairseq-py/examples/mr_hubert/config/decode \
  --config-name infer \
  task.data=/path/to/data \
  task.normalize=[true|false] \
  decoding.exp_dir=/path/to/experiment/directory \
  common_eval.path=/path/to/checkpoint \
  dataset.gen_subset=test
```

### 2. KenLM Decoding (with n-gram language model)

```bash
python examples/speech_recognition/new/infer.py \
  --config-dir /path/to/fairseq-py/examples/mr_hubert/config/decode \
  --config-name infer_lm \
  task.data=/path/to/data \
  task.normalize=[true|false] \
  decoding.exp_dir=/path/to/experiment/directory \
  common_eval.path=/path/to/checkpoint \
  dataset.gen_subset=test \
  decoding.decoder.lexicon=/path/to/lexicon \
  decoding.decoder.lmpath=/path/to/arpa
```

**Important decoding parameters:**
- `decoding.decoder.beam`: Beam size (default varies)
- `decoding.decoder.beamthreshold`: Beam pruning threshold
- `decoding.decoder.lmweight`: Language model weight
- `decoding.decoder.wordscore`: Word insertion score
- `decoding.decoder.silweight`: Silence weight

### 3. Fairseq-LM Decoding (with neural language model)

Similar to KenLM decoding but with a neural LM (refer to wav2vec2/hubert examples for details).