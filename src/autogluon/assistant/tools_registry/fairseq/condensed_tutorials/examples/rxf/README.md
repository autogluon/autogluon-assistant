# Condensed: [Better Fine-Tuning by Reducing Representational Collapse](https://arxiv.org/abs/2008.03156)

Summary: This tutorial implements R3F and R4F methods for reducing representational collapse during model fine-tuning. It provides code for sentence prediction tasks with noise-augmented training, featuring customizable noise distribution parameters (normal or uniform), control over noise-KL loss weighting, and optional spectral normalization. The implementation extends Fairseq with custom criteria and can be applied to language models like RoBERTa. The tutorial includes complete command-line examples showing how to fine-tune models with these techniques, highlighting key hyperparameters and configuration options that improve model generalization through noise-robust representations.

*This is a condensed version that preserves essential implementation details and context.*

# Better Fine-Tuning by Reducing Representational Collapse

## Implementation Details

This repository implements the R3F and R4F methods from the paper "Better Fine-Tuning by Reducing Representational Collapse".

- **R3F**: Registered as `sentence_prediction_r3f`
- **Label smoothing version**: Implemented as `label_smoothed_cross_entropy_r3f`
- **R4F**: Use R3F with spectral normalization via `--spectral-norm-classification-head` parameter

## Key Hyperparameters

Three critical hyperparameters:
- `--eps`: Standard deviation/range of the noise distribution
- `--r3f-lambda`: Controls the weight between logistic loss and noisy KL loss
- `--noise-type`: Distribution type ('normal' or 'uniform')

## Example Usage

```bash
TOTAL_NUM_UPDATES=3120
WARMUP_UPDATES=187
LR=1e-05
NUM_CLASSES=2
MAX_SENTENCES=8        # Batch size.
ROBERTA_PATH=/path/to/roberta/model.pt

CUDA_VISIBLE_DEVICES=0 fairseq-train RTE-bin \
    --restore-file $ROBERTA_PATH \
    --max-positions 512 \
    --max-sentences $MAX_SENTENCES \
    --max-tokens 4400 \
    --task sentence_prediction \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --init-token 0 --separator-token 2 \
    --arch roberta_large \
    --criterion sentence_prediction_r3f \
    --num-classes $NUM_CLASSES \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
    --max-epoch 10 \
    --find-unused-parameters \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --noise-type uniform --r3f-lambda 0.7 \
    --user-dir examples/rxf/rxf_src
```

Note: The implementation requires adding the custom modules via `--user-dir examples/rxf/rxf_src`