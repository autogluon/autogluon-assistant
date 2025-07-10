# Condensed: Non-autoregressive Neural Machine Translation (NAT)

Summary: This tutorial provides implementation guidance for Non-autoregressive Neural Machine Translation (NAT) models, particularly the Levenshtein Transformer. It covers knowledge distillation techniques essential for NAT training, dataset preparation with WMT'14 En-De, and detailed training configurations with fairseq. Key functionalities include iterative refinement generation, advanced decoding methods (ensemble decoding, length-beam techniques), and re-ranking with autoregressive models. The tutorial offers specific command-line parameters for training and inference, noise application techniques, and optimization strategies—valuable for developers implementing efficient machine translation systems that balance quality and inference speed.

*This is a condensed version that preserves essential implementation details and context.*

# Non-autoregressive Neural Machine Translation (NAT)

## Implementation Overview

This tutorial covers implementation of several NAT models, primarily focusing on the Levenshtein Transformer and knowledge distillation techniques for NAT.

## Dataset Preparation

1. Follow the WMT'14 En-De dataset preprocessing instructions with joined dictionary:
   ```bash
   fairseq-preprocess --joined-dictionary
   ```

2. **Knowledge Distillation**: Essential for NAT models to learn good translations
   - Train a standard transformer model on the same data
   - Decode the training set to produce a distillation dataset
   - Pre-processed datasets available: [original](http://dl.fbaipublicfiles.com/nat/original_dataset.zip) and [distillation](http://dl.fbaipublicfiles.com/nat/distill_dataset.zip)

## Training Levenshtein Transformer

```bash
fairseq-train \
    data-bin/wmt14_en_de_distill \
    --save-dir checkpoints \
    --ddp-backend=legacy_ddp \
    --task translation_lev \
    --criterion nat_loss \
    --arch levenshtein_transformer \
    --noise random_delete \
    --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9,0.98)' \
    --lr 0.0005 --lr-scheduler inverse_sqrt \
    --stop-min-lr '1e-09' --warmup-updates 10000 \
    --warmup-init-lr '1e-07' --label-smoothing 0.1 \
    --dropout 0.3 --weight-decay 0.01 \
    --decoder-learned-pos \
    --encoder-learned-pos \
    --apply-bert-init \
    --log-format 'simple' --log-interval 100 \
    --fixed-validation-seed 7 \
    --max-tokens 8000 \
    --save-interval-updates 10000 \
    --max-update 300000
```

Key parameters:
- `--task translation_lev`: Task for Levenshtein Transformer
- `--criterion nat_loss`: NAT-specific loss function
- `--noise random_delete`: Input noise type for target sentences

## Translation Generation

Uses `iterative_refinement_generator` which iteratively refines translations:

```bash
fairseq-generate \
    data-bin/wmt14_en_de_distill \
    --gen-subset test \
    --task translation_lev \
    --path checkpoints/checkpoint_best.pt \
    --iter-decode-max-iter 9 \
    --iter-decode-eos-penalty 0 \
    --beam 1 --remove-bpe \
    --print-step \
    --batch-size 400
```

Important parameters:
- `--iter-decode-max-iter`: Maximum iterations for refinement
- `--iter-decode-eos-penalty`: Penalty to prevent generating too short translations (typically 0-3)
- `--print-step`: Shows actual iterations per sentence

## Advanced Decoding Methods

### Ensemble Decoding
```bash
fairseq-generate \
    data-bin/wmt14_en_de_distill \
    --gen-subset test \
    --task translation_lev \
    --path checkpoint_1.pt:checkpoint_2.pt:checkpoint_3.pt \
    --iter-decode-max-iter 9 \
    --iter-decode-eos-penalty 0 \
    --beam 1 --remove-bpe \
    --print-step \
    --batch-size 400
```
Use `:` to separate multiple model paths.

### Length-beam
For models that predict lengths before decoding (vanilla NAT, Mask-Predict), varying target lengths can improve quality.
- Not applicable to models that dynamically change lengths (Insertion/Levenshtein Transformer)

### Re-ranking with Autoregressive Model
```bash
fairseq-generate \
    data-bin/wmt14_en_de_distill \
    --gen-subset test \
    --task translation_lev \
    --path checkpoints/checkpoint_best.pt:at_checkpoints/checkpoint_best.pt \
    --iter-decode-max-iter 9 \
    --iter-decode-eos-penalty 0 \
    --iter-decode-with-beam 9 \
    --iter-decode-with-external-reranker \
    --beam 1 --remove-bpe \
    --print-step \
    --batch-size 100
```

**Important**: Ensure the autoregressive model shares the same vocabulary as the NAT model.