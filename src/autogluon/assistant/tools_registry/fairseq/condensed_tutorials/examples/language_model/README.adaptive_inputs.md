# Condensed: Adaptive Input Representations for Neural Language Modeling (Baevski and Auli, 2018)

Summary: This tutorial demonstrates implementing adaptive input representations for neural language modeling using fairseq. It provides pre-trained models for Google Billion Words and WikiText-103 datasets, along with complete training code. The implementation features the transformer_lm_wiki103 architecture, adaptive loss criterion, cosine learning rate scheduling, and NAG optimization. This resource helps with building efficient language models that use adaptive input embeddings to reduce parameters while maintaining performance, particularly useful for implementing state-of-the-art language modeling techniques with varying embedding sizes across vocabulary frequency bins.

*This is a condensed version that preserves essential implementation details and context.*

# Adaptive Input Representations for Neural Language Modeling

## Pre-trained Models

| Description | Parameters | Dataset | Model Link |
|-------------|----------:|---------|------------|
| Adaptive Inputs (Baevski and Auli, 2018) | 1026M | Google Billion Words | [download](https://dl.fbaipublicfiles.com/fairseq/models/lm/adaptive_lm_gbw_huge.tar.bz2) |
| Adaptive Inputs (Baevski and Auli, 2018) | 247M | WikiText-103 | [download](https://dl.fbaipublicfiles.com/fairseq/models/lm/adaptive_lm_wiki103.v2.tar.bz2) |

## Training Implementation

To train a model with adaptive inputs on WikiText-103:

```bash
fairseq-train --task language_modeling \
    data-bin/wikitext-103 \
    --save-dir checkpoints/transformer_wikitext-103 \
    --arch transformer_lm_wiki103 \
    --max-update 286000 
    --lr 1.0 
    --t-mult 2 
    --lr-period-updates 270000 
    --lr-scheduler cosine 
    --lr-shrink 0.75 \
    --warmup-updates 16000 
    --warmup-init-lr 1e-07 
    --stop-min-lr 1e-09 
    --optimizer nag 
    --min-lr 0.0001 
    --clip-norm 0.1 \
    --criterion adaptive_loss 
    --max-tokens 3072 
    --update-freq 3 
    --tokens-per-sample 3072 
    --seed 1 \
    --sample-break-mode none 
    --skip-invalid-size-inputs-valid-test 
    --ddp-backend=legacy_ddp
```

Key configuration details:
- Uses `transformer_lm_wiki103` architecture
- Learning rate: 1.0 with cosine scheduler
- Criterion: `adaptive_loss`
- Tokens per sample: 3072
- Uses NAG optimizer with gradient clipping at 0.1

Refer to the general language modeling README for WikiText-103 preprocessing instructions.