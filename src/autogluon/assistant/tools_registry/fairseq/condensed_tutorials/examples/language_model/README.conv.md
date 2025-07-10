# Condensed: Language Modeling with Gated Convolutional Networks (Dauphin et al., 2017)

Summary: This tutorial demonstrates how to implement and evaluate a gated convolutional language model using the fairseq library. It provides the complete command-line configuration for training the fconv_lm_dauphin_wikitext103 architecture on the WikiText-103 dataset, including specific hyperparameters like adaptive softmax cutoffs, dropout settings, and optimization parameters. The tutorial covers model training with NAG optimizer, learning rate scheduling, and token handling configurations. It also shows how to evaluate the trained language model. This knowledge is valuable for implementing convolutional language models, configuring fairseq for language modeling tasks, and understanding key hyperparameters for optimal performance.

*This is a condensed version that preserves essential implementation details and context.*

# Language Modeling with Gated Convolutional Networks

## Implementation

To train a convolutional language model using the `fconv_lm_dauphin_wikitext103` architecture:

```bash
fairseq-train --task language_modeling \
    data-bin/wikitext-103 \
    --save-dir checkpoints/fconv_wikitext-103 \
    --arch fconv_lm_dauphin_wikitext103 \
    --adaptive-softmax-cutoff 10000,20000,200000 \
    --dropout 0.2 \
    --criterion adaptive_loss \
    --optimizer nag --clip-norm 0.1 --weight-decay 5e-06 \
    --lr 1.0 --lr-scheduler reduce_lr_on_plateau --lr-shrink 0.5 \
    --max-tokens 1024 --tokens-per-sample 1024 \
    --ddp-backend legacy_ddp \
    --max-epoch 35
```

## Evaluation

```bash
fairseq-eval-lm data-bin/wikitext-103 --path checkpoints/fconv_wiki103/checkpoint_best.pt
```

## Key Configuration Parameters
- Architecture: `fconv_lm_dauphin_wikitext103`
- Adaptive softmax cutoffs: 10000, 20000, 200000
- Dropout: 0.2
- Optimizer: NAG with clip-norm 0.1, weight decay 5e-06
- Learning rate: 1.0 with reduce-on-plateau scheduler and 0.5 shrink factor
- Tokens: 1024 max tokens and tokens per sample