# Condensed: Simultaneous Machine Translation

Summary: This tutorial implements Monotonic Multihead Attention for simultaneous machine translation, featuring three different approaches: MMA-IL (Infinite Lookback), MMA-H (Hard Aligned), and Wait-k models. It provides complete training commands for each implementation using fairseq, with specific configuration parameters for latency control (using different weight parameters like latency-weight-avg and latency-weight-var). The tutorial covers data preparation for WMT'15 En-De and En-Ja datasets, and demonstrates how to configure transformer architectures with appropriate optimization settings for simultaneous translation tasks where translation begins before the full source sentence is available.

*This is a condensed version that preserves essential implementation details and context.*

# Simultaneous Machine Translation

Implementation of [Monotonic Multihead Attention](https://openreview.net/forum?id=Hyg96gBKPS) paper.

## Data Preparation
Follow instructions to download and preprocess the WMT'15 En-De dataset from the [fairseq translation example](https://github.com/pytorch/fairseq/tree/simulastsharedtask/examples/translation#prepare-wmt14en2desh).

For English-Japanese implementation, see [enja.md](docs/enja.md).

## Training Models

### MMA-IL (Infinite Lookback)
```shell
fairseq-train \
    data-bin/wmt15_en_de_32k \
    --simul-type infinite_lookback \
    --user-dir $FAIRSEQ/example/simultaneous_translation \
    --mass-preservation \
    --criterion latency_augmented_label_smoothed_cross_entropy \
    --latency-weight-avg 0.1 \
    --max-update 50000 \
    --arch transformer_monotonic_iwslt_de_en save_dir_key=lambda \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr-scheduler 'inverse_sqrt' \
    --warmup-init-lr 1e-7 --warmup-updates 4000 \
    --lr 5e-4 --stop-min-lr 1e-9 --clip-norm 0.0 --weight-decay 0.0001 \
    --dropout 0.3 \
    --label-smoothing 0.1 \
    --max-tokens 3584
```

### MMA-H (Hard Aligned)
```shell
fairseq-train \
    data-bin/wmt15_en_de_32k \
    --simul-type hard_aligned \
    --user-dir $FAIRSEQ/example/simultaneous_translation \
    --mass-preservation \
    --criterion latency_augmented_label_smoothed_cross_entropy \
    --latency-weight-var 0.1 \
    --max-update 50000 \
    --arch transformer_monotonic_iwslt_de_en save_dir_key=lambda \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr-scheduler 'inverse_sqrt' \
    --warmup-init-lr 1e-7 --warmup-updates 4000 \
    --lr 5e-4 --stop-min-lr 1e-9 --clip-norm 0.0 --weight-decay 0.0001 \
    --dropout 0.3 \
    --label-smoothing 0.1 \
    --max-tokens 3584
```

### Wait-k Model
```shell
fairseq-train \
    data-bin/wmt15_en_de_32k \
    --simul-type wait-k \
    --waitk-lagging 3 \
    --user-dir $FAIRSEQ/example/simultaneous_translation \
    --mass-preservation \
    --criterion latency_augmented_label_smoothed_cross_entropy \
    --max-update 50000 \
    --arch transformer_monotonic_iwslt_de_en save_dir_key=lambda \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr-scheduler 'inverse_sqrt' \
    --warmup-init-lr 1e-7 --warmup-updates 4000 \
    --lr 5e-4 --stop-min-lr 1e-9 --clip-norm 0.0 --weight-decay 0.0001 \
    --dropout 0.3 \
    --label-smoothing 0.1 \
    --max-tokens 3584
```

Key differences between implementations:
- MMA-IL uses `--latency-weight-avg 0.1`
- MMA-H uses `--latency-weight-var 0.1`
- Wait-k uses `--waitk-lagging 3`