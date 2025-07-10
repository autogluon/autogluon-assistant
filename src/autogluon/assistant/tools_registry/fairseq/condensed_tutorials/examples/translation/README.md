# Condensed: Neural Machine Translation

Summary: This tutorial demonstrates neural machine translation implementation using PyTorch/fairseq, covering both pre-trained model usage and custom model training. It shows how to load and use models via torch.hub for interactive translation, perform CLI-based translation with BLEU scoring, and train new transformer and convolutional models on IWSLT and WMT datasets. The guide includes complete code for data preprocessing, model training with hyperparameters, and multilingual translation setup. Key functionalities include batch translation, model evaluation, checkpoint management, and handling multiple language pairs with shared decoders—essential knowledge for implementing production-ready neural translation systems.

*This is a condensed version that preserves essential implementation details and context.*

# Neural Machine Translation Implementation Guide

## Using Pre-trained Models with torch.hub

### Setup
```bash
pip install fastBPE sacremoses subword_nmt
```

### Interactive Translation
```python
import torch

# Load a transformer model
en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt16.en-de',
                       tokenizer='moses', bpe='subword_nmt')
en2de.eval()  # disable dropout
en2de.cuda()  # move to GPU

# Translate text
en2de.translate('Hello world!')  # 'Hallo Welt!'

# Batch translation
en2de.translate(['Hello world!', 'The cat sat on the mat.'])
```

### Loading Custom Models
```python
from fairseq.models.transformer import TransformerModel
zh2en = TransformerModel.from_pretrained(
  '/path/to/checkpoints',
  checkpoint_file='checkpoint_best.pt',
  data_name_or_path='data-bin/wmt17_zh_en_full',
  bpe='subword_nmt',
  bpe_codes='data-bin/wmt17_zh_en_full/zh.code'
)
```

### WMT19 Models (FastBPE)
```python
en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de',
                       checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt',
                       tokenizer='moses', bpe='fastbpe')
```

## CLI Usage

### Generate Translations
```bash
fairseq-generate data-bin/wmt14.en-fr.newstest2014 \
    --path data-bin/wmt14.en-fr.fconv-py/model.pt \
    --beam 5 --batch-size 128 --remove-bpe
```

### Score BLEU
```bash
grep ^H /tmp/gen.out | cut -f3- > /tmp/gen.out.sys
grep ^T /tmp/gen.out | cut -f2- > /tmp/gen.out.ref
fairseq-score --sys /tmp/gen.out.sys --ref /tmp/gen.out.ref
```

## Training New Models

### IWSLT'14 German to English (Transformer)

1. **Preprocess data**:
```bash
# Download and prepare
bash prepare-iwslt14.sh

# Binarize
fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt14.tokenized.de-en \
    --workers 20
```

2. **Train model**:
```bash
fairseq-train data-bin/iwslt14.tokenized.de-en \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric
```

### WMT'14 English to German (Convolutional)

1. **Preprocess data**:
```bash
# Download and prepare
bash prepare-wmt14en2de.sh  # Use --icml17 for original paper settings

# Binarize
fairseq-preprocess \
    --source-lang en --target-lang de \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/wmt17_en_de --thresholdtgt 0 --thresholdsrc 0 \
    --workers 20
```

2. **Train model**:
```bash
fairseq-train data-bin/wmt17_en_de \
    --arch fconv_wmt_en_de \
    --dropout 0.2 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --optimizer nag --clip-norm 0.1 \
    --lr 0.5 --lr-scheduler fixed --force-anneal 50 \
    --max-tokens 4000
```

## Multilingual Translation

### Training a {de,fr}-en Model

1. **Install dependencies**:
```bash
pip install sacrebleu sentencepiece
```

2. **Preprocess data**:
```bash
# Download and prepare
bash prepare-iwslt17-multilingual.sh

# Binarize German-English
fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $TEXT/train.bpe.de-en \
    --validpref $TEXT/valid0.bpe.de-en,$TEXT/valid1.bpe.de-en,$TEXT/valid2.bpe.de-en,$TEXT/valid3.bpe.de-en,$TEXT/valid4.bpe.de-en,$TEXT/valid5.bpe.de-en \
    --destdir data-bin/iwslt17.de_fr.en.bpe16k \
    --workers 10

# Binarize French-English (reuse English dictionary)
fairseq-preprocess --source-lang fr --target-lang en \
    --trainpref $TEXT/train.bpe.fr-en \
    --validpref $TEXT/valid0.bpe.fr-en,$TEXT/valid1.bpe.fr-en,$TEXT/valid2.bpe.fr-en,$TEXT/valid3.bpe.fr-en,$TEXT/valid4.bpe.fr-en,$TEXT/valid5.bpe.fr-en \
    --tgtdict data-bin/iwslt17.de_fr.en.bpe16k/dict.en.txt \
    --destdir data-bin/iwslt17.de_fr.en.bpe16k \
    --workers 10
```

3. **Train multilingual model**:
```bash
fairseq-train data-bin/iwslt17.de_fr.en.bpe16k/ \
    --task multilingual_translation --lang-pairs de-en,fr-en \
    --arch multilingual_transformer_iwslt_de_en \
    --share-decoders --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr 0.0005 --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 --warmup-init-lr '1e-07' \
    --label-smoothing 0.1 --criterion label_smoothed_cross_entropy \
    --dropout 0.3 --weight-decay 0.0001 \
    --max-tokens 4000 \
    --update-freq 8  # Simulates 8 GPUs with gradient accumulation
```

**Important**: During inference, specify `--source-lang` and `--target-lang` while keeping `--lang-pairs` the same as during training.