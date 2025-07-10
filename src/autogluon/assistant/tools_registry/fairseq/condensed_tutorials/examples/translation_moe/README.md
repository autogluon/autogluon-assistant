# Condensed: Mixture Models for Diverse Machine Translation: Tricks of the Trade (Shen et al., 2019)

Summary: This tutorial provides implementation guidance for Mixture of Experts (MoE) models in machine translation using Fairseq. It covers techniques for training diverse translation models with multiple experts (hard or soft mixtures with learned/uniform priors). Key functionalities include data preparation with WMT datasets, MoE model training with various parameterization options, expert-specific translation generation, and evaluation metrics for translation diversity (pairwise BLEU, reference coverage, multi-reference BLEU). The code helps with implementing translation systems that produce diverse outputs rather than a single translation, using shared parameterization and online responsibility assignment.

*This is a condensed version that preserves essential implementation details and context.*

# Mixture Models for Diverse Machine Translation: Implementation Guide

## Data Preparation
- Use WMT'17 En-De dataset
- Ensure joint vocabulary with `--joined-dictionary` option during preprocessing

## Training MoE Models

```bash
fairseq-train --ddp-backend='legacy_ddp' \
    data-bin/wmt17_en_de \
    --max-update 100000 \
    --task translation_moe --user-dir examples/translation_moe/translation_moe_src \
    --method hMoElp --mean-pool-gating-network \
    --num-experts 3 \
    --arch transformer_wmt_en_de --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
    --lr 0.0007 \
    --dropout 0.1 --weight-decay 0.0 --criterion cross_entropy \
    --max-tokens 3584
```

### Key Parameters
- `--method`: MoE variant options
  - `hMoElp`: Hard mixture with learned prior
  - `hMoEup`: Hard mixture with uniform prior
  - `sMoElp`: Soft mixture with learned prior
  - `sMoEup`: Soft mixture with uniform prior
- `--num-experts`: Number of experts (3 in example)
- Uses online responsibility assignment and shared parameterization

## Generation

Generate translations from a specific expert:
```bash
fairseq-generate data-bin/wmt17_en_de \
    --path checkpoints/checkpoint_best.pt \
    --beam 1 --remove-bpe \
    --task translation_moe --user-dir examples/translation_moe/translation_moe_src \
    --method hMoElp --mean-pool-gating-network \
    --num-experts 3 \
    --gen-expert 0
```

## Evaluation Process

1. Download reference data:
```bash
wget dl.fbaipublicfiles.com/fairseq/data/wmt14-en-de.extra_refs.tok
```

2. Generate translations from each expert:
```bash
BPE_CODE=examples/translation/wmt17_en_de/code
for EXPERT in $(seq 0 2); do \
    cat wmt14-en-de.extra_refs.tok \
    | grep ^S | cut -f 2 \
    | fairseq-interactive data-bin/wmt17_en_de \
        --path checkpoints/checkpoint_best.pt \
        --beam 1 \
        --bpe subword_nmt --bpe-codes $BPE_CODE \
        --buffer-size 500 --max-tokens 6000 \
        --task translation_moe --user-dir examples/translation_moe/translation_moe_src \
        --method hMoElp --mean-pool-gating-network \
        --num-experts 3 \
        --gen-expert $EXPERT ; \
done > wmt14-en-de.extra_refs.tok.gen.3experts
```

3. Calculate metrics:
```bash
python examples/translation_moe/score.py --sys wmt14-en-de.extra_refs.tok.gen.3experts --ref wmt14-en-de.extra_refs.tok
```

Expected results (matching Table 7, row 3):
- Pairwise BLEU: 48.26
- References covered: 2.11
- Multi-reference BLEU: 59.46