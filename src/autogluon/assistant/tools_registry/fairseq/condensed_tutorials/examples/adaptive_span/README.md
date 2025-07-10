# Condensed: Adaptive Span

Summary: This tutorial implements Adaptive Span, a self-attention mechanism that dynamically learns optimal attention spans to handle extended context while controlling computational costs. It provides complete code for training and evaluating Adaptive Span models on the Enwik8 dataset using fairseq, including specific command-line parameters for model configuration. Key features include configurable attention spans (up to 8192 tokens), auxiliary loss scaling to control span length, and performance comparison with TransformerXL models. The implementation demonstrates how to achieve state-of-the-art language modeling results (~1.03 bpc on test) while efficiently managing memory usage through adaptive attention spans.

*This is a condensed version that preserves essential implementation details and context.*

# Adaptive Span Implementation

Adaptive Span is a self-attention mechanism that learns its optimal attention span, allowing for extended context size while controlling memory and computational costs.

## Setup

Process the Enwik8 dataset:
```bash
fairseq-preprocess --only-source --trainpref ~/data/enwik8/train.txt \
    --validpref ~/data/enwik8/valid.txt --testpref ~/data/enwik8/test.txt \
    --destdir ~/data/enwik8/data-bin/ --joined-dictionary --workers 20
```

## Training

Train a 12-layer Adaptive Span model (4 GPUs, total batch size 64):
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-train \
    --user-dir examples/adaptive_span \
    --data ~/data/enwik8/data-bin/ \
    --fp16 --fp16-no-flatten-grads --max-update 600000 \
    --task truncated_bptt_lm --tokens-per-sample 512 --arch adaptive_span \
    --n-layer 12 --d-model 512 --n-head 8 --d-inner 2048 --dropout 0.3 \
    --attn-span 8192 --optimizer adagrad_with_grad_clip --adagrad-clip 0.03 \
    --validate-interval-updates 1000 \
    --lr-scheduler fixed --warmup-updates 32000 --batch-size-valid 32 \
    --lr 0.07 --criterion adaptive_span_loss --batch-size 16 --update-freq 1 \
    --seed 2 --log-format json --log-interval 25 --aux-loss-scaler 5e-07
```

**Key Parameters:**
- `--attn-span 8192`: Maximum attention span
- `--aux-loss-scaler 5e-07`: Controls span length (lower for better performance but longer span)
- `--update-freq 4`: For single GPU training to simulate 4 GPUs

**Expected Results:** ~1.05 on validation, ~1.03 on test

## TransformerXL Alternative

For comparison, train a TransformerXL model:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-train \
    --user-dir examples/truncated_bptt \
    ~/data/enwik8/data-bin/ \
    --task truncated_bptt_lm --fp16 --max-update 400000 \
    --tokens-per-sample 512 --arch transformer_xl --n-layer 12 \
    --d-model 512 --n-head 8 --d-head 64 --d-inner 2048 --dropout 0.1 \
    --dropatt 0.0 --mem-len 512 --optimizer adam --clip-norm 0.25 \
    --lr-scheduler cosine --warmup-updates 0 \
    --lr 0.00025 --batch-size 15 \
    --update-freq 1 --seed 2 --log-format json --log-interval 25
```

**Expected Results:** ~1.06 on test

## Evaluation

For Adaptive Span:
```bash
fairseq-eval-lm ~/data/enwik8/data-bin/ --path model/checkpoint_best.pt \
 --user-dir examples/adaptive_span \
 --task truncated_bptt_lm --batch-size 8 --tokens-per-sample 512 --gen-subset test
```

For TransformerXL:
```bash
fairseq-eval-lm ~/data/enwik8/data-bin/ --path model/checkpoint_best.pt \
    --user-dir examples/truncated_bptt/ --task truncated_bptt_lm --batch-size 8 \
    --tokens-per-sample 80 \
    --model-overrides '{"mem_len":2100,"clamp_len":820,"same_length":True}' \
    --gen-subset valid
```

**Note:** Training context was 512 tokens with batch size 8, matching the evaluation settings from the original paper.