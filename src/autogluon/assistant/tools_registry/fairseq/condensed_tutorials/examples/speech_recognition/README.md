# Condensed: 2021 Update: We are merging this example into the [S2T framework](../speech_to_text), which supports more generic speech-to-text tasks (e.g. speech translation) and more flexible data processing pipelines. Please stay tuned.

Summary: This tutorial covers implementing Automatic Speech Recognition (ASR) in Fairseq, based on the "Transformers with convolutional context for ASR" paper. It provides implementation knowledge for training ASR models with VGG-Transformer architectures and Flashlight integration for Conv/GLU models with ASG loss. The tutorial helps with data preparation (using LibriSpeech), model training, inference with beam search, and evaluation using WER metrics. Key features include n-gram language model integration, lexicon formatting for different decoders, and support for both character-based and word-piece targets, with code examples for each step of the ASR pipeline.

*This is a condensed version that preserves essential implementation details and context.*

# Speech Recognition in Fairseq

## Overview
This module implements ASR tasks in Fairseq based on the paper [Transformers with convolutional context for ASR](https://arxiv.org/abs/1904.11660).

> **2021 Update:** This example is being merged into the S2T framework which supports more generic speech-to-text tasks.

## Dependencies
- [torchaudio](https://github.com/pytorch/audio) - for audio feature extraction
- [Sclite](http://www1.icsi.berkeley.edu/Speech/docs/sctk-1.2/sclite.htm) - for WER measurement
- [sentencepiece](https://github.com/google/sentencepiece) - for word-piece targets

## Data Preparation
```bash
./examples/speech_recognition/datasets/prepare-librispeech.sh $DIR_TO_SAVE_RAW_DATA $DIR_FOR_PREPROCESSED_DATA
```

## Training
```bash
python train.py $DIR_FOR_PREPROCESSED_DATA --save-dir $MODEL_PATH \
  --max-epoch 80 --task speech_recognition --arch vggtransformer_2 \
  --optimizer adadelta --lr 1.0 --adadelta-eps 1e-8 --adadelta-rho 0.95 \
  --clip-norm 10.0 --max-tokens 5000 --log-format json --log-interval 1 \
  --criterion cross_entropy_acc --user-dir examples/speech_recognition/
```

## Inference
```bash
python examples/speech_recognition/infer.py $DIR_FOR_PREPROCESSED_DATA \
  --task speech_recognition --max-tokens 25000 --nbest 1 \
  --path $MODEL_PATH/checkpoint_last.pt --beam 20 --results-path $RES_DIR \
  --batch-size 40 --gen-subset $SET --user-dir examples/speech_recognition/
```

## Evaluation
```bash
sclite -r ${RES_DIR}/ref.word-checkpoint_last.pt-${SET}.txt \
  -h ${RES_DIR}/hypo.word-checkpoint_last.pt-${SET}.txt -i rm -o all stdout > $RES_REPORT
```

## Flashlight Integration

### Setup
Install flashlight python bindings from [flashlight v0.3.2](https://github.com/flashlight/flashlight/tree/e16682fa32df30cbf675c8fe010f929c61e3b833/bindings/python).

### Training with Conv/GLU + ASG Loss
```bash
python train.py $DIR_FOR_PREPROCESSED_DATA --save-dir $MODEL_PATH \
  --max-epoch 100 --task speech_recognition --arch w2l_conv_glu_enc \
  --batch-size 4 --optimizer sgd --lr 0.3,0.8 --momentum 0.8 \
  --clip-norm 0.2 --max-tokens 50000 --log-format json --log-interval 100 \
  --num-workers 0 --sentence-avg --criterion asg_loss \
  --asg-transitions-init 5 --max-replabel 2 --linseg-updates 8789 \
  --user-dir examples/speech_recognition
```

> **Important:** ASG loss works better with character targets. Set `nbpe=31` in `prepare-librispeech.sh`.

### Inference with n-gram LM
```bash
python examples/speech_recognition/infer.py $DIR_FOR_PREPROCESSED_DATA \
  --task speech_recognition --seed 1 --nbest 1 \
  --path $MODEL_PATH/checkpoint_last.pt --gen-subset $SET \
  --results-path $RES_DIR --w2l-decoder kenlm --kenlm-model $KENLM_MODEL_PATH \
  --lexicon $LEXICON_PATH --beam 200 --beam-threshold 15 \
  --lm-weight 1.5 --word-score 1.5 --sil-weight -0.3 \
  --criterion asg_loss --max-replabel 2 --user-dir examples/speech_recognition
```

#### Lexicon Format
For ASG inference:
```
doorbell  D O 1 R B E L 1 ▁
```

For CTC with word-pieces:
```
doorbell  ▁DOOR BE LL
doorbell  ▁DOOR B E LL
```

> **Note:** The *word* should match the case of the n-gram LM, while the *spelling* should match the case of the token dictionary.

### Viterbi-only Inference
```bash
python examples/speech_recognition/infer.py $DIR_FOR_PREPROCESSED_DATA \
  --task speech_recognition --seed 1 --nbest 1 \
  --path $MODEL_PATH/checkpoint_last.pt --gen-subset $SET \
  --results-path $RES_DIR --w2l-decoder viterbi \
  --criterion asg_loss --max-replabel 2 --user-dir examples/speech_recognition
```