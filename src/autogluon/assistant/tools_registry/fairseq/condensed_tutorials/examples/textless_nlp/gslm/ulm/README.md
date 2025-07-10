# Condensed: Unit Language Model (ULM)

Summary: This tutorial provides implementation knowledge for Unit Language Models (ULMs) using fairseq, covering data preprocessing, model training, and sampling. It helps with tasks like training transformer-based language models on discrete speech units and generating speech unit sequences. Key features include configuring transformer architectures (6 or 12 layers), training parameters optimization, memory management techniques, conditional/unconditional sampling with temperature control, and working with pre-trained ULMs for various unit types (LogMel, CPC, HuBERT, Wav2Vec 2.0) with different vocabulary sizes.

*This is a condensed version that preserves essential implementation details and context.*

# Unit Language Model (ULM)

## Pre-trained Models
Pre-trained ULMs are available for various unit types (LogMel Filterbank, Modified CPC, HuBERT, Wav2Vec 2.0) with different vocabulary sizes (50, 100, 200). Download links are provided in the original documentation.

## Implementation Details

### Preprocessing Data
```bash
fairseq-preprocess --only-source \
    --trainpref data/train.txt --validpref data/valid.txt --testpref data/test.txt \
    --destdir data-bin/ --workers 40
```

### Training ULM
```bash
fairseq-train data-bin/ \
    --task=language_modeling \
    --arch=transformer_lm_big \
    --share-decoder-input-output-embed \
    --dropout=0.1 \
    --attention-dropout=0.1 \
    --optimizer=adam \
    --adam-betas='(0.9, 0.98)' \
    --clip-norm=1.0 \
    --lr=0.0005 \
    --lr-scheduler=inverse_sqrt \
    --warmup-updates=4000 \
    --warmup-init-lr=1e-07 \
    --tokens-per-sample=3072 \
    --update-freq=16 \
    --max-tokens=4096 \
    --num-workers=4 \
    --skip-invalid-size-inputs-valid-test \
    --max-update=500000 \
    --log-interval=10 \
    --seed=100501 \
    --fp16 \
    --sample-break-mode=eos
```

**Key Configuration Notes:**
- Default is Transformer-large (12 layers)
- Use `--arch=transformer_lm` for smaller 6-layer model
- Adjust `--update-freq` when using different number of GPUs
- Enable `--checkpoint-activations` to save GPU memory at expense of computation

### Sampling from ULM
```bash
python sample.py data-bin/ \
    --path=checkpoints/checkpoint_best.pt \
    --task=language_modeling \
    --sampling --temperature=0.7 \
    --seed=1 \
    --prompts=prompts.txt \
    --output=samples.txt \
    --max-len-a=0 --max-len-b=500 \
    --prefix-size=-1 \
    --batch-size=16 \
    --fp16 \
    --samples-per-prompt=10
```

**Important Parameters:**
- `--prefix-size`: Controls tokens used to prime the ULM
  - `-1`: Use entire prompt
  - `0`: Unconditional sampling
  - Positive value: Use first n tokens
- `--samples-per-prompt`: Number of utterances generated per prompt
- `--max-len-a` and `--max-len-b`: Control number of generated tokens

For pre-trained models, `data-bin` should point to the unpacked directory containing `dict.txt`.