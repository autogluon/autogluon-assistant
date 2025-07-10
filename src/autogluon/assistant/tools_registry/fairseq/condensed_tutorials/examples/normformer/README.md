# Condensed: NormFormer

Summary: This tutorial implements NormFormer, a technique that enhances transformer models by adding strategic normalization layers. It provides specific command-line flags (--scale-attn, --scale-fc, --scale-heads, --scale-resids) to modify existing fairseq-train commands for different model sizes (125M to 2.7B parameters). The implementation helps with optimizing transformer training by explaining how to adjust learning rates, batch sizes, and update frequencies based on model size. Key features include model-specific configurations, technical requirements for FSDP training, and guidance on adapting the approach to different datasets while maintaining effective batch sizes.

*This is a condensed version that preserves essential implementation details and context.*

# NormFormer Implementation Guide

## Overview
Implementation of ["NormFormer: Improved Transformer Pretraining with Extra Normalization"](https://arxiv.org/abs/2110.09456) which adds strategic normalization layers to transformer models.

## Key Implementation Details

### Core NormFormer Flags
To convert any existing `fairseq-train` command to use NormFormer, add:
```bash
--scale-attn --scale-fc --scale-heads
```

For smaller models, also consider adding:
```bash
--scale-resids
```

### Best Practices
- Increase learning rate when using NormFormer
- Adjust batch size parameters to maintain effective batch size:
  - 125M and 355M models: (1024×1024×0.5) tokens
  - 1.3B+ models: (1024×1024) tokens
- Set `--update-freq` = 256/`global_bs` for small models or 512/`global_bs` for large models
  - Where `global_bs` = `--batch-size` * `--distributed-world-size`

## Model-Specific Configurations

### 125M Parameters
```bash
# Best configuration
train_125M --lr 3e-3 --scale-attn --scale-fc --scale-heads --scale-resids
```

### 355M Parameters
```bash
# Best configuration
train_355M --lr 1e-3 --scale-attn --scale-fc --scale-heads --scale-resids
```

### 1.3B Parameters
```bash
# NormFormer configuration
train_1.3B --lr 6e-4 --scale-attn --scale-fc --scale-heads
```

### 2.7B Parameters
```bash
# NormFormer configuration
train_2.7B --lr 6e-4 --activation-fn relu_squared --scale-attn --scale-fc --scale-heads
```

## Technical Requirements
- Requires `fairscale>=0.4.0` for FSDP (Fully Sharded Data Parallel)
- Small models can train on as few as 8 GPUs
- Preprocessing follows standard fairseq language model preprocessing

## Important Note
NormFormer results in the paper use a larger private dataset. For best results, adapt preprocessing to your dataset and compare to a baseline on the same data.