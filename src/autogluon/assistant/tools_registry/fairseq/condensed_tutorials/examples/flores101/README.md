# Condensed: <p align="center">

Summary: This tutorial demonstrates implementing multilingual machine translation using Flores101 models (175M or 615M parameters). It covers downloading pretrained models supporting 100+ languages, preprocessing text with SentencePiece tokenization, binarizing data with fairseq-preprocess, and generating translations using fairseq-generate. Key features include handling a 256K vocabulary, using language tokens for encoder/decoder, FP16 inference optimization, beam search translation, and the translation_multi_simple_epoch task. The implementation is particularly useful for multilingual translation tasks involving both high and low-resource languages.

*This is a condensed version that preserves essential implementation details and context.*

# Flores101: Large-Scale Multilingual Machine Translation

## Pretrained Models

| Model | Layers | Embed dim | FFN dim | Vocab Size | Params | Download |
|---|---|---|---|---|---|---|
| `flores101_mm100_615M` | 12 | 1024 | 4096 | 256,000 | 615M | [Download](https://dl.fbaipublicfiles.com/flores101/pretrained_models/flores101_mm100_615M.tar.gz) |
| `flores101_mm100_175M` | 6 | 512 | 2048 | 256,000 | 175M | [Download](https://dl.fbaipublicfiles.com/flores101/pretrained_models/flores101_mm100_175M.tar.gz) |

These models are trained similar to [M2M-100](https://arxiv.org/abs/2010.11125) with additional language support for the WMT Large-Scale Multilingual Translation track.

## Implementation Guide

### 1. Download and Extract Model

```bash
fairseq=/path/to/fairseq
cd $fairseq

# Download 615M param model
wget https://dl.fbaipublicfiles.com/flores101/pretrained_models/flores101_mm100_615M.tar.gz

# Extract 
tar -xvzf flores101_mm100_615M.tar.gz
```

### 2. Encode with SentencePiece

```bash
# Download example dataset (German to French)
sacrebleu --echo src -l de-fr -t wmt19 | head -n 20 > raw_input.de-fr.de
sacrebleu --echo ref -l de-fr -t wmt19 | head -n 20 > raw_input.de-fr.fr

for lang in de fr ; do
    python scripts/spm_encode.py \
        --model flores101_mm100_615M/sentencepiece.bpe.model \
        --output_format=piece \
        --inputs=raw_input.de-fr.${lang} \
        --outputs=spm.de-fr.${lang}
done
```

### 3. Binarize Data

```bash
fairseq-preprocess \
    --source-lang de --target-lang fr \
    --testpref spm.de-fr \
    --thresholdsrc 0 --thresholdtgt 0 \
    --destdir data_bin \
    --srcdict flores101_mm100_615M/dict.txt --tgtdict flores101_mm100_615M/dict.txt
```

### 4. Generate Translations

```bash
fairseq-generate \
    data_bin \
    --batch-size 1 \
    --path flores101_mm100_615M/model.pt \
    --fixed-dictionary flores101_mm100_615M/dict.txt \
    -s de -t fr \
    --remove-bpe 'sentencepiece' \
    --beam 5 \
    --task translation_multi_simple_epoch \
    --lang-pairs flores101_mm100_615M/language_pairs.txt \
    --decoder-langtok --encoder-langtok src \
    --gen-subset test \
    --fp16 \
    --dataset-impl mmap \
    --distributed-world-size 1 --distributed-no-spawn
```

## Key Implementation Details

- Models support 100+ languages (full list in original documentation)
- Uses SentencePiece tokenization with a 256K vocabulary
- Requires language tokens for both encoder and decoder
- Supports FP16 inference for faster generation
- Uses beam search with beam size 5 by default
- Implements the `translation_multi_simple_epoch` task for multilingual translation

The model supports 101 languages including major languages like English, Chinese, Arabic, Russian, and many low-resource languages like Aymara, Chokwe, and Umbundu.