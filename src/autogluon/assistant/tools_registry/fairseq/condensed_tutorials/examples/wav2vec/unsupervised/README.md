# Condensed: wav2vec Unsupervised  (wav2vec-U)

Summary: This tutorial implements wav2vec-U and wav2vec-U 2.0 frameworks for unsupervised speech recognition without labeled data. It covers three key processes: (1) data preparation techniques including silence removal, feature extraction from audio, and text/phoneme processing; (2) GAN-based training with configurable hyperparameters for learning speech-text alignments; and (3) iterative self-training with Kaldi LM-decoding. The implementation provides scripts for audio preprocessing, feature extraction, phoneme generation, and model training. Developers can use this to build speech recognition systems for low-resource languages, generate transcriptions using various decoding strategies, and implement unsupervised speech-to-text pipelines.

*This is a condensed version that preserves essential implementation details and context.*

# wav2vec Unsupervised (wav2vec-U) Implementation Guide

## Overview
Wav2vec-U and wav2vec-U 2.0 are frameworks for building speech recognition systems without labeled training data. The training process consists of three main steps:
1. Preparation of speech representations and text data
2. Generative adversarial training (GAN)
3. Iterative self-training with Kaldi LM-decoding

## Prerequisites
```bash
# Set environment variables
export FAIRSEQ_ROOT=/path/to/fairseq
export RVAD_ROOT=/path/to/rVADfast
export KENLM_ROOT=/path/to/kenlm/binaries
export KALDI_ROOT=/path/to/kaldi
```

## Data Preparation

### Audio Processing
1. Remove silence from audio:
```bash
# Create manifest file
python $FAIRSEQ_ROOT/examples/wav2vec/wav2vec_manifest.py /audio/dir --ext wav --dest /path/to/train.tsv --valid-percent 0

# Generate VAD segments
python scripts/vads.py -r $RVAD_ROOT < /path/to/train.tsv > train.vads

# Remove silence
python scripts/remove_silence.py --tsv /path/to/train.tsv --vads train.vads --out /output/audio/dir

# Create new manifest
python $FAIRSEQ_ROOT/examples/wav2vec/wav2vec_manifest.py /output/audio/dir --ext wav --dest /path/to/new/train.tsv --valid-percent 0.01
```

2. Preprocess audio features:
```bash
# For wav2vec-U
zsh scripts/prepare_audio.sh /dir/with/tsv /output/dir /path/to/wav2vec2/model.pt 512 14

# For wav2vec-U 2.0
zsh scripts/prepare_audio_v2.sh /dir/with/tsv /output/dir /path/to/wav2vec2/model.pt 64 14
```
**Note**: The third argument is PCA dimensionality (wav2vec-U) or MFCC clusters (wav2vec-U 2.0). The last argument is the layer index (0-based) for feature extraction.

### Text Processing
```bash
zsh scripts/prepare_text.sh language /path/to/text/file /output/dir 1000 espeak /path/to/fasttext/lid/model sil_prob
```
**Parameters**:
- Fourth argument: minimum observations of phones to keep (reduce for small corpora)
- Fifth argument: phonemizer to use (espeak, espeak-ng, or G2P for English)
- Last argument: silence probability between words (0.25 for wav2vec-U, 0.5 for 2.0)

### TIMIT Data Preparation
```bash
bash scripts/prepare_timit.sh /dir/to/timit/raw/data /output/dir /path/to/wav2vec2/model.pt
```

## GAN Training

```bash
PREFIX=w2v_unsup_gan_xp

# For wav2vec-U (pre-segmented features)
CONFIG_NAME=w2vu
TASK_DATA=/path/to/features/precompute_unfiltered_pca512_cls128_mean_pooled

# For wav2vec-U 2.0 (raw features)
CONFIG_NAME=w2vu2
TASK_DATA=/path/to/features/

# Text data and language model
TEXT_DATA=/path/to/data/phones
KENLM_PATH=/path/to/data/phones/kenlm.phn.o4.bin

PYTHONPATH=$FAIRSEQ_ROOT PREFIX=$PREFIX fairseq-hydra-train \
    -m --config-dir config/gan \
    --config-name $CONFIG_NAME \
    task.data=${TASK_DATA} \
    task.text_data=${TEXT_DATA} \
    task.kenlm_path=${KENLM_PATH} \
    common.user_dir=${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised \
    model.code_penalty=2,4 model.gradient_penalty=1.5,2.0 \
    model.smoothness_weight=0.5,0.75,1.0 'common.seed=range(0,5)'
```

## Generating Transcriptions

```bash
python w2vu_generate.py --config-dir config/generate --config-name viterbi \
fairseq.common.user_dir=${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised \
fairseq.task.data=/path/to/dir/with/features \
fairseq.common_eval.path=/path/to/gan/checkpoint \ 
fairseq.dataset.gen_subset=valid results_path=/where/to/save/transcriptions
```

**Important notes**:
- Decoding without LM works best on adjacent-mean-pooled features
- Decoding with LM works better on features before mean-pooling
- For wav2vec-U 2.0, add `decode_stride=1` or `2` for better results

## Iterative Self-Training

After GAN training provides the initial unsupervised model:
1. Pseudo-label training data with the GAN model
2. Train an HMM on the pseudo-labels
3. Relabel training data with the HMM
4. Fine-tune the original wav2vec 2.0 model using HMM pseudo-labels with CTC loss

Refer to `kaldi_self_train/README.md` for detailed instructions on iterative self-training and Kaldi LM-decoding.