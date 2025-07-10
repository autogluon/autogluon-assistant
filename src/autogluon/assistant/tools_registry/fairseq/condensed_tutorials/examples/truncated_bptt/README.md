# Condensed: Truncated Backpropagation Through Time (BPTT)

Summary: This tutorial demonstrates implementing Truncated Backpropagation Through Time (BPTT) for training language models on long sequences, as used in Transformer-XL. It covers sequential data iteration with gradient flow limited to current chunks while conditioning on previous chunks, supporting batching and multi-GPU training. Key functionalities include customizing the FairseqTask class, training Transformer-XL on WikiText-103 with specific architecture parameters (16 layers, 410 embedding dimensions, 150-token context windows), and evaluation techniques that use extended memory contexts. The implementation enables efficient training on long sequences while maintaining contextual information across chunks.

*This is a condensed version that preserves essential implementation details and context.*

# Truncated Backpropagation Through Time (BPTT) Implementation

## Overview
Truncated BPTT enables training language models on very long sequences by splitting them into chunks and conditioning on previous chunks while only flowing gradients through the current chunk. This technique was used in Transformer-XL to achieve state-of-the-art results.

## Implementation Details
- Overrides `FairseqTask::get_batch_iterator` to iterate over data sequentially
- Supports batching and multi-GPU (data parallel) training
- Disables batch shuffling logic

## Training Transformer-XL on WikiText-103

### Setup
First, preprocess WikiText-103 data following the language modeling README.

### Training Command
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-train \
    --user-dir examples/truncated_bptt \
    data-bin/wikitext-103/ \
    --task truncated_bptt_lm --tokens-per-sample 150 \
    --batch-size 15 --max-update 200000 \
    --arch transformer_xl --n-layer 16 --d-model 410 --n-head 10 \
    --d-head 41 --d-inner 2100 --dropout 0.1 --dropatt 0.0 --mem-len 150 \
    --optimizer adam --clip-norm 0.25 \
    --lr-scheduler cosine --warmup-updates 0 --min-lr 0.0 --lr 0.00025  \
    --log-format json --log-interval 25 \
    --fp16
```

**Key Parameters:**
- `--tokens-per-sample 150`: Context length
- `--mem-len 150`: Memory length for previous context
- `--n-layer 16`: 16-layer Transformer-XL
- `--d-model 410`, `--n-head 10`, `--d-head 41`, `--d-inner 2100`: Model dimensions

**Note:** For single GPU training, use `--update-freq=4` to simulate 4 GPUs.

## Evaluation

```bash
fairseq-eval-lm data-bin/wikitext-103/ \
    --path checkpoints/checkpoint_best.pt \
    --user-dir examples/truncated_bptt/ \
    --task truncated_bptt_lm \
    --batch-size 1 --required-batch-size-multiple 1 \
    --model-overrides '{"mem_len":640,"clamp_len":400,"same_length":True}' \
    --tokens-per-sample 64
```

**Evaluation Settings:**
- Training: 150 tokens context + 150 memory tokens
- Evaluation: 64 tokens per sample with increased memory length (640)
- Expected result: ~23.70 perplexity (paper reported 24.0)