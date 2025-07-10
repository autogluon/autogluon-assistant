# Condensed: Multilingual Translation

Summary: This tutorial provides implementation knowledge for multilingual translation using fairseq, covering temperature-based sampling for unbalanced datasets, language token configuration, and mBART model finetuning. It helps with preprocessing multilingual data (using joint BPE vocabularies), training translation models with specific parameters, finetuning pretrained mBART50 models, and generating/evaluating translations. Key features include handling multiple language pairs simultaneously, configurable language tokens for source/target sentences, temperature-based dataset sampling, and integration with pretrained mBART50 models (many-to-one, one-to-many, and many-to-many variants).

*This is a condensed version that preserves essential implementation details and context.*

# Multilingual Translation

## Key Implementation Details

This framework supports multilingual translation with multiple bitext datasets, featuring:

- Temperature-based sampling over unbalanced datasets
- Configurable language token addition to source/target sentences
- Finetuning capabilities for mBART pretrained models

## Preprocessing

Requires a joint BPE vocabulary:
- Follow [mBART's preprocessing steps](https://github.com/pytorch/fairseq/tree/main/examples/mbart#bpe-data) to reuse pretrained sentence-piece model
- Or train your own joint BPE model

## Training

```bash
fairseq-train $path_2_data \
  --encoder-normalize-before --decoder-normalize-before \
  --arch transformer --layernorm-embedding \
  --task translation_multi_simple_epoch \
  --sampling-method "temperature" \
  --sampling-temperature 1.5 \
  --encoder-langtok "src" \
  --decoder-langtok \
  --lang-dict "$lang_list" \
  --lang-pairs "$lang_pairs" \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
  --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
  --lr-scheduler inverse_sqrt --lr 3e-05 --warmup-updates 2500 --max-update 40000 \
  --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.0 \
  --max-tokens 1024 --update-freq 2 \
  --save-interval 1 --save-interval-updates 5000 --keep-interval-updates 10 --no-epoch-checkpoints \
  --seed 222 --log-format simple --log-interval 2
```

## Finetuning

Finetune from pretrained models like mMBART:

```bash
fairseq-train $path_2_data \
  --finetune-from-model $pretrained_model \
  # Same parameters as training command above
```

## Generation

```bash
fairseq-generate $path_2_data \
  --path $model \
  --task translation_multi_simple_epoch \
  --gen-subset test \
  --source-lang $source_lang \
  --target-lang $target_lang \
  --sacrebleu --remove-bpe 'sentencepiece' \
  --batch-size 32 \
  --encoder-langtok "src" \
  --decoder-langtok \
  --lang-dict "$lang_list" \
  --lang-pairs "$lang_pairs"
```

For custom evaluation:
```bash
cat {source_lang}_${target_lang}.txt | grep -P "^H" |sort -V |cut -f 3- |$TOK_CMD > ${source_lang}_${target_lang}.hyp
cat {source_lang}_${target_lang}.txt | grep -P "^T" |sort -V |cut -f 2- |$TOK_CMD > ${source_lang}_${target_lang}.ref
sacrebleu -tok 'none' -s 'none' ${source_lang}_${target_lang}.ref < ${source_lang}_${target_lang}.hyp
```

## mBART50 Models

Available pretrained models:
- [mMBART 50 pretrained model](https://dl.fbaipublicfiles.com/fairseq/models/mbart50/mbart50.pretrained.tar.gz)
- [mMBART 50 finetuned many-to-one](https://dl.fbaipublicfiles.com/fairseq/models/mbart50/mbart50.ft.n1.tar.gz)
- [mMBART 50 finetuned one-to-many](https://dl.fbaipublicfiles.com/fairseq/models/mbart50/mbart50.ft.1n.tar.gz)
- [mMBART 50 finetuned many-to-many](https://dl.fbaipublicfiles.com/fairseq/models/mbart50/mbart50.ft.nn.tar.gz)

Each tarball contains:
- Model checkpoint: `model.pt`
- Supported languages list: `ML50_langs.txt`
- Sentence piece model: `sentence.bpe.model`
- Dictionaries for each language: `dict.{lang}.txt`

Usage:
1. Binarize data using `binarize.py` with the sentence.bpe.model and dictionaries
2. Run generation command with appropriate parameters