# Condensed: Fully Sharded Data Parallel (FSDP)

Summary: This tutorial explains Fully Sharded Data Parallel (FSDP) implementation in Fairseq, a technique that shards model parameters and optimizer state across GPUs to enable training of large language models. It covers how to configure FSDP with options like `--ddp-backend=fully_sharded`, `--cpu-offload`, and `--no-reshard-after-forward`. The tutorial demonstrates practical code examples for training 13B parameter models on both single and multiple GPUs, highlighting compatibility with pointwise optimizers, CPU offloading requirements, and performance optimizations. Key functionalities include mixed precision training, checkpoint activations, and integration with DeepSpeed's CPU Adam optimizer for memory-efficient training of large language models.

*This is a condensed version that preserves essential implementation details and context.*

# Fully Sharded Data Parallel (FSDP) in Fairseq

## Overview

FSDP is an efficient data parallel training approach that shards model parameters and optimizer state across data parallel workers. Compared to PyTorch DDP:

- Produces identical results (synchronous data parallel training)
- Shards parameters (FP16 + FP32) and optimizer state across GPUs
- Faster due to sharded optimizer step and overlapped communication
- Enables training 13B parameter models on 8 GPUs and 175B parameter models on 128 GPUs

## Key Implementation Options

```bash
--ddp-backend=fully_sharded    # enables full sharding via FSDP
--cpu-offload                  # offloads optimizer state and FP32 model copy to CPU
--no-reshard-after-forward     # increases speed for large models (1B+ params), similar to ZeRO stage 2
```

Standard options like `--fp16`, `--update-freq`, `--checkpoint-activations` continue to work normally.

## Implementation Details

### Limitations
- Compatible with pointwise Optimizers (Adam, AdamW, SGD) but not with non-pointwise ones (Adagrad, Adafactor, LAMB)
- Models requiring `--fp16-no-flatten-grads` may not be supported

### Training Large Models

#### 13B params on 1 V100 GPU (with CPU offloading)

**Requirements:**
- Latest fairscale: `pip install git+https://github.com/facebookresearch/fairscale.git@master`
- 32GB GPU memory and ~256GB system memory
- DeepSpeed for CPU Adam: `pip install deepspeed`

**Key configuration:**
```bash
OMP_NUM_THREADS=20 CUDA_VISIBLE_DEVICES=0 \
    fairseq-train data-bin/wikitext-103-roberta-bpe-bin \
    --ddp-backend fully_sharded --fp16 --fp16-init-scale 4 \
    --cpu-offload --checkpoint-activations \
    --task language_modeling --tokens-per-sample 2048 --batch-size 8 \
    --arch transformer_lm_gpt3_13 \
    --optimizer cpu_adam --adam-betas "(0.9,0.98)" \
    --lr 0.0001 --lr-scheduler polynomial_decay --warmup-updates 5 --total-num-update 10
```

**Important notes:**
- Initialization takes ~5 minutes for 13B parameters
- CPU offloading requires mixed precision (`--fp16`)
- Tune `OMP_NUM_THREADS` for best performance

#### 13B params on 8 V100 GPUs (with full parameter + optimizer state sharding)

For best performance on 8 GPUs, combine full sharding with CPU offloading:

```bash
OMP_NUM_THREADS=20 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    fairseq-train data-bin/wikitext-103-roberta-bpe-bin \
    --ddp-backend fully_sharded --fp16 --fp16-init-scale 4 \
    --cpu-offload --checkpoint-activations \
    --task language_modeling --tokens-per-sample 2048 --batch-size 8 \
    --arch transformer_lm_gpt3_13 \
    --optimizer cpu_adam --adam-betas "(0.9,0.98)" \
    --lr 0.0001 --lr-scheduler polynomial_decay --warmup-updates 5 --total-num-update 10
```

**Performance improvement:** Training speed increases superlinearly from ~310 words per second (1 GPU) to ~3200 words per second (8 GPUs).