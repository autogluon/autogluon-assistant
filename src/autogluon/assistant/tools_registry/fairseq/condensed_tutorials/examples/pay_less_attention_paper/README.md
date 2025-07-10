# Condensed: Pay Less Attention with Lightweight and Dynamic Convolutions (Wu et al., 2019)

Summary: This tutorial implements "Pay Less Attention with Lightweight and Dynamic Convolutions" paper, featuring LightConv and DynamicConv models as alternatives to attention mechanisms in sequence modeling. It provides memory-efficient CUDA kernels that save ~50% memory, pre-trained models for multiple language pairs (with/without GLUs), and code for model loading via torch.hub or custom paths. The tutorial covers implementation details for training configurations across different translation tasks (IWSLT14, WMT16/17), including hyperparameters, model averaging, and evaluation. Key functionalities include GLU integration options, lightweight/dynamic convolution selection, and multi-GPU training support.

*This is a condensed version that preserves essential implementation details and context.*

# Pay Less Attention with Lightweight and Dynamic Convolutions

## Implementation Overview

This document covers pre-trained models and implementation details for the paper "Pay Less Attention with Lightweight and Dynamic Convolutions" (Wu et al., 2019).

## Pre-trained Models

Models are available with and without GLUs (Gated Linear Units), with the non-GLU versions being faster at inference:

- **IWSLT14 German-English**: LightConv and DynamicConv models (without GLUs)
- **WMT16 English-German**: Models available with and without GLUs
- **WMT14 English-French**: Models with GLUs
- **WMT17 Chinese-English**: Models with GLUs

## Memory-Efficient CUDA Kernels

The repository provides memory-efficient CUDA kernels that save ~50% memory compared to PyTorch implementations for large sequence lengths:

```sh
# Install lightconv
cd fairseq/modules/lightconv_layer
python cuda_function_gen.py
python setup.py install

# Install dynamicconv
cd fairseq/modules/dynamicconv_layer
python cuda_function_gen.py
python setup.py install
```

## Usage with torch.hub

```python
import torch

# Load a model
zh2en = torch.hub.load('pytorch/fairseq', 'lightconv.glu.wmt17.zh-en', 
                      tokenizer='moses', bpe='subword_nmt')

# Translate
zh2en.translate('你好 世界')  # 'Hello World'
```

## Loading Custom Models

```python
from fairseq.models.lightconv import LightConvModel
en2fr = LightConvModel.from_pretrained(
  '/path/to/checkpoints',
  checkpoint_file='checkpoint_best.pt',
  data_name_or_path='data-bin/wmt14_en_fr',
  bpe='subword_nmt',
  bpe_codes='data-bin/wmt14_en_fr/en.code'
)
```

## Training Configuration

### Key Parameters
- To use model without GLU: `--encoder-glu 0 --decoder-glu 0`
- For LightConv: `--encoder-conv-type lightweight --decoder-conv-type lightweight`
- Default is DynamicConv if not specified

### IWSLT14 De-En Training Example

```sh
SAVE="save/dynamic_conv_iwslt"
mkdir -p $SAVE 
CUDA_VISIBLE_DEVICES=0 fairseq-train data-bin/iwslt14.tokenized.de-en \
    --clip-norm 0 --optimizer adam --lr 0.0005 \
    --source-lang de --target-lang en --max-tokens 4000 \
    --log-interval 100 --stop-min-lr '1e-09' --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --lr-scheduler inverse_sqrt \
    --ddp-backend=legacy_ddp \
    --max-update 50000 --warmup-updates 4000 --warmup-init-lr '1e-07' \
    --adam-betas '(0.9, 0.98)' --keep-last-epochs 10 \
    -a lightconv_iwslt_de_en --save-dir $SAVE \
    --dropout 0.3 --attention-dropout 0.1 --weight-dropout 0.1 \
    --encoder-glu 0 --decoder-glu 0
```

### Model Averaging and Evaluation

```sh
# Average last 10 checkpoints
python scripts/average_checkpoints.py --inputs $SAVE \
    --num-epoch-checkpoints 10 --output "${SAVE}/checkpoint_last10_avg.pt"

# Evaluation
fairseq-generate data-bin/iwslt14.tokenized.de-en \
    --path "${SAVE}/checkpoint_last10_avg.pt" --batch-size 128 \
    --beam 4 --remove-bpe --lenpen 1 --gen-subset test
```

### WMT16 En-De Training (Multi-GPU)

```sh
SAVE="save/dynamic_conv_wmt16en2de"
mkdir -p $SAVE
python -m torch.distributed.launch --nproc_per_node 8 $(which fairseq-train) \
    data-bin/wmt16_en_de_bpe32k --fp16 --log-interval 100 \
    --max-update 30000 --share-all-embeddings --optimizer adam \
    --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --stop-min-lr 1e-09 --update-freq 16 --attention-dropout 0.1 \
    --ddp-backend=legacy_ddp --max-tokens 3584 \
    --lr-scheduler cosine --warmup-init-lr 1e-7 --warmup-updates 10000 \
    --lr 0.001 --min-lr 1e-7 --t-mult 1 --lr-period-updates 20000 \
    --arch lightconv_wmt_en_de_big --save-dir $SAVE \
    --dropout 0.3 --attention-dropout 0.1 --weight-dropout 0.1 \
    --encoder-glu 1 --decoder-glu 1
```

For WMT models, the cosine scheduler is used with specific hyperparameters for learning rate and warmup.