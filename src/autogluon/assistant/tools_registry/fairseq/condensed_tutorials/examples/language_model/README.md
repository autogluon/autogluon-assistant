# Condensed: Neural Language Modeling

Summary: This tutorial covers neural language modeling with fairseq, demonstrating implementation of transformer-based language models. It provides code for using pre-trained models via PyTorch Hub (loading models, sampling text, computing perplexity) and CLI-based training workflows. Key functionalities include: preprocessing text data, training transformer language models with adaptive inputs, evaluating models using perplexity, and handling context windows. The tutorial features practical code examples for both inference with pre-trained models and training custom models, with specific parameters for optimization, memory management, and performance tuning.

*This is a condensed version that preserves essential implementation details and context.*

# Neural Language Modeling

## Pre-trained Models

| Model | Description | Dataset | Download |
|---|---|---|---|
| `transformer_lm.gbw.adaptive_huge` | Adaptive Inputs (1026M params) | Google Billion Words | [download](https://dl.fbaipublicfiles.com/fairseq/models/lm/adaptive_lm_gbw_huge.tar.bz2) |
| `transformer_lm.wiki103.adaptive` | Adaptive Inputs (247M params) | WikiText-103 | [download](https://dl.fbaipublicfiles.com/fairseq/models/lm/adaptive_lm_wiki103.v2.tar.bz2) |
| `transformer_lm.wmt19.en/de/ru` | WMT19 Language Models | WMT News Crawl | [en](https://dl.fbaipublicfiles.com/fairseq/models/lm/wmt19.en.tar.gz) / [de](https://dl.fbaipublicfiles.com/fairseq/models/lm/wmt19.de.tar.gz) / [ru](https://dl.fbaipublicfiles.com/fairseq/models/lm/wmt19.ru.tar.gz) |

## Usage with PyTorch Hub

```bash
pip install fastBPE sacremoses  # Required dependencies
```

```python
import torch

# Load model
en_lm = torch.hub.load('pytorch/fairseq', 'transformer_lm.wmt19.en', 
                      tokenizer='moses', bpe='fastbpe')
en_lm.eval()  # disable dropout
en_lm.cuda()  # move to GPU

# Sample from model
en_lm.sample('Barack Obama', beam=1, sampling=True, 
            sampling_topk=10, temperature=0.8)

# Compute perplexity
en_lm.score('Barack Obama is coming to Sydney and New Zealand')['positional_scores'].mean().neg().exp()

# Load custom model
from fairseq.models.transformer_lm import TransformerLanguageModel
custom_lm = TransformerLanguageModel.from_pretrained('/path/to/model/dir', 
                                                   'checkpoint100.pt', 
                                                   tokenizer='moses', 
                                                   bpe='fastbpe')
```

## Training with CLI Tools

### 1) Preprocess Data

```bash
# Download and prepare WikiText-103
cd examples/language_model/
bash prepare-wikitext-103.sh
cd ../..

# Binarize data
TEXT=examples/language_model/wikitext-103
fairseq-preprocess \
    --only-source \
    --trainpref $TEXT/wiki.train.tokens \
    --validpref $TEXT/wiki.valid.tokens \
    --testpref $TEXT/wiki.test.tokens \
    --destdir data-bin/wikitext-103 \
    --workers 20
```

### 2) Train Language Model

```bash
fairseq-train --task language_modeling \
  data-bin/wikitext-103 \
  --save-dir checkpoints/transformer_wikitext-103 \
  --arch transformer_lm --share-decoder-input-output-embed \
  --dropout 0.1 \
  --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
  --tokens-per-sample 512 --sample-break-mode none \
  --max-tokens 2048 --update-freq 16 \
  --fp16 \
  --max-update 50000
```

**Note:** If you run out of memory, try reducing `--max-tokens` or `--tokens-per-sample`. Adjust `--update-freq` to simulate training on different number of GPUs.

### 3) Evaluate

```bash
fairseq-eval-lm data-bin/wikitext-103 \
    --path checkpoints/transformer_wiki103/checkpoint_best.pt \
    --batch-size 2 \
    --tokens-per-sample 512 \
    --context-window 400
```

**Context Window Note:** The `--context-window` parameter controls context provided for perplexity calculation:
- `0`: Dataset chunked into segments of length 512, higher perplexity due to less conditioning
- Maximum value (511): Each token fully conditioned on 511 tokens of context, slower but better perplexity

## Additional Resources
For convolutional language models, see the convolutional LM README.