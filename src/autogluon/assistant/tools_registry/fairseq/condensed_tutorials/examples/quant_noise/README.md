# Condensed: Training with Quantization Noise for Extreme Model Compression ({Fan\*, Stock\*} *et al.*, 2020)

Summary: This tutorial explains implementing Quantization Noise for model compression using two techniques: Scalar Quantization and Iterative Product Quantization (iPQ). It covers how to train models with quantization noise by adding specific command-line flags, integration steps for both techniques, and the quantization process workflow. The guide provides practical code examples for integrating quantization into existing models, complete training commands for RoBERTa and language models, and quantization procedures. Key functionalities include controlling noise proportion during training, automatic quantization during evaluation, combining with LayerDrop for further compression, and finetuning centroids after quantization for optimal performance.

*This is a condensed version that preserves essential implementation details and context.*

# Training with Quantization Noise for Model Compression

This guide covers implementation details for training and quantizing models with Quantization Noise, supporting both scalar quantization and Iterative Product Quantization (iPQ).

## Scalar Quantization Implementation

Scalar quantization with Quant-Noise randomly quantizes a proportion `p` of weights during training using Fake Quantization (emulating int8 on GPU).

### Training
```bash
--quant-noise-scalar 0.5
```
Higher noise values make networks easier to quantize but may increase non-quantized perplexity.

### Quantization
During evaluation, all modules automatically switch to `p=1` (fully quantized).

### Integration
To use in your code:
- Use `quantize_model_` from `fairseq/modules/quantization/scalar/utils.py` to replace modules with quantized versions and add activation hooks
- In `eval()` mode, the network is fully quantized by default

## Iterative Product Quantization (iPQ)

### Training
Add these flags:
```bash
--quant-noise-pq 0.1 --quant-noise-pq-block-size 8
```

Recommendations:
- Use 0.05-0.2 for `quant-noise-pq`
- Use block-size of 8 (must be multiple of `input_features`)
- Can combine with LayerDrop (0.1-0.2) for additional model size reduction

### Quantization Process
1. Perform ~20 steps of Product Quantization
2. Finetune the resulting centroids

### Integration
```python
from fairseq.modules.quantization.pq import quantize_model_, SizeTracker

# get configuration parameters
n_centroids_config = config["n_centroids"]
block_sizes_config = config["block_sizes"]
layers_to_quantize = config["layers_to_quantize"]

# size tracker for keeping track of assignments, centroids and non-compressed sizes
size_tracker = SizeTracker(model)

# Quantize model by stages
for step in range(len(layers_to_quantize)):
    # quantize model in-place
    quantized_layers = quantize_model_(
        model,
        size_tracker,
        layers_to_quantize,
        block_sizes_config,
        n_centroids_config,
        step=step,
    )
    logger.info(f"Finetuning stage {step}, quantized layers: {quantized_layers}")
    logger.info(f"{size_tracker}")

    # Re-create/update trainer/optimizer since model parameters have changed
    optimizer = ...

    # Finetune the centroids with your usual training loop
    trainer.train_epoch()
```

## Reproducing NLP Results

### RoBERTa + QuantNoise Training
```bash
fairseq-train $DATA_DIR \
    --task masked_lm --criterion masked_lm --arch roberta_base \
    --sample-break-mode complete \
    --tokens-per-sample 512 --max-positions 512 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-6 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr 0.0005 \
    --warmup-updates 10000 --total-num-update 125000 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 \
    --batch-size 16 \
    --update-freq 2 --max-update 125000 \
    --save-dir checkpoint/roberta \
    --ddp-backend legacy_ddp --encoder-layerdrop 0.2 \
    --quant-noise-pq 0.2 --quant-noise-pq-block-size 8 --untie-weights-roberta
```

### Language Model Training on Wikitext-103
```bash
fairseq-train --task language_modeling /path/to/wikitext-103/data \
    --save-dir checkpoints/transformer_wikitext-103 \
    --adaptive-input --adaptive-input-cutoff 20000,60000 --adaptive-input-factor 4 \
    --adaptive-softmax-cutoff 20000,60000 --adaptive-softmax-dropout 0.2 --adaptive-softmax-factor 4.0 \
    --tie-adaptive-proj --tie-adaptive-weights \
    --arch transformer_lm_gbw \
    --attention-dropout 0.1 --dropout 0.2 --relu-dropout 0.1 \
    --clip-norm 0.1 --criterion adaptive_loss \
    --ddp-backend legacy_ddp \
    --decoder-attention-heads 8 --decoder-embed-dim 1024 --decoder-ffn-embed-dim 4096 --decoder-input-dim 1024 \
    --decoder-layers 16 --decoder-normalize-before --decoder-output-dim 1024 \
    --min-lr 0.0001 --lr-period-updates 270000 --lr-scheduler cosine --lr-shrink 0.75 --lr 1.0 --t-mult 2.0 \
    --max-tokens 3072 --tokens-per-sample 3072 --momentum 0.99 --optimizer nag \
    --sample-break-mode none --update-freq 3 \
    --warmup-init-lr 1e-07 --warmup-updates 16000 \
    --weight-decay 0 --seed 1 --stop-min-lr 1e-09 \
    --quant-noise-pq 0.05 --quant-noise-pq-block-size 8
```

### Quantizing Models
For RoBERTa (runs on 1 GPU):
```bash
fairseq-train --task sentence_prediction /path/to/data/ \
    --restore-file $ROBERTA_PATH \
    --save-dir checkpoints/roberta_finetuned \
    --max-positions 512 \
    --batch-size 16 \
    --max-tokens 4400 \
    --init-token 0 --separator-token 2 \
    --arch roberta_large \
    --criterion sentence_prediction \
    --num-classes 2 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 --lr-scheduler polynomial_decay \
    --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
    --no-progress-bar --skip-invalid-size-inputs-valid-test --ddp-backend legacy_ddp \
    --quantization-config-path /path/to/config/yaml
```

## Notes
- Scalar quantization with RoBERTa and combined iPQ+int8 quantization have not been fully tested
- Vision model code will be released as part of ClassyVision