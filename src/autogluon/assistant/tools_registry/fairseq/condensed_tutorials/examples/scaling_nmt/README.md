# Condensed: Scaling Neural Machine Translation (Ott et al., 2018)

Summary: This tutorial demonstrates implementing neural machine translation using the Fairseq library, focusing on transformer models for English-French and English-German translation. It covers: (1) downloading pre-trained models, (2) data preprocessing with joined dictionaries, (3) training transformer models with specific hyperparameters and optimization techniques including mixed precision (FP16), and (4) model evaluation through checkpoint averaging and BLEU score calculation. Key features include large-scale training optimizations, multi-GPU simulation via update frequency adjustment, and both compound split and detokenized BLEU evaluation methods. This resource helps with implementing state-of-the-art NMT systems based on the Ott et al. (2018) paper.

*This is a condensed version that preserves essential implementation details and context.*

# Scaling Neural Machine Translation (Ott et al., 2018)

## Pre-trained Models

| Model | Dataset | Download |
|---|---|---|
| `transformer.wmt14.en-fr` | WMT14 English-French | [model](https://dl.fbaipublicfiles.com/fairseq/models/wmt14.en-fr.joined-dict.transformer.tar.bz2), [newstest2014](https://dl.fbaipublicfiles.com/fairseq/data/wmt14.en-fr.joined-dict.newstest2014.tar.bz2) |
| `transformer.wmt16.en-de` | WMT16 English-German | [model](https://dl.fbaipublicfiles.com/fairseq/models/wmt16.en-de.joined-dict.transformer.tar.bz2), [newstest2014](https://dl.fbaipublicfiles.com/fairseq/data/wmt16.en-de.joined-dict.newstest2014.tar.bz2) |

## Training on WMT'16 En-De

### 1. Extract the data
```bash
TEXT=wmt16_en_de_bpe32k
mkdir -p $TEXT
tar -xzvf wmt16_en_de.tar.gz -C $TEXT
```

### 2. Preprocess with joined dictionary
```bash
fairseq-preprocess \
    --source-lang en --target-lang de \
    --trainpref $TEXT/train.tok.clean.bpe.32000 \
    --validpref $TEXT/newstest2013.tok.bpe.32000 \
    --testpref $TEXT/newstest2014.tok.bpe.32000 \
    --destdir data-bin/wmt16_en_de_bpe32k \
    --nwordssrc 32768 --nwordstgt 32768 \
    --joined-dictionary \
    --workers 20
```

### 3. Train the model
```bash
fairseq-train \
    data-bin/wmt16_en_de_bpe32k \
    --arch transformer_vaswani_wmt_en_de_big --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
    --dropout 0.3 --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 3584 \
    --fp16
```

**IMPORTANT:** For better performance:
- Add `--update-freq 16` to simulate training on 128 GPUs (with 8 GPUs)
- Increase learning rate to 0.001 for large batches
- `--fp16` requires CUDA 9.1+ and Volta GPU or newer

### 4. Evaluate

#### Average checkpoints
```bash
python scripts/average_checkpoints \
    --inputs /path/to/checkpoints \
    --num-epoch-checkpoints 10 \
    --output checkpoint.avg10.pt
```

#### Generate translations
```bash
fairseq-generate \
    data-bin/wmt16_en_de_bpe32k \
    --path checkpoint.avg10.pt \
    --beam 4 --lenpen 0.6 --remove-bpe > gen.out
```

#### Compute BLEU scores
Compound split BLEU (not recommended):
```bash
bash scripts/compound_split_bleu.sh gen.out
```

Detokenized BLEU with sacrebleu (preferred):
```bash
bash scripts/sacrebleu.sh wmt14/full en de gen.out
```