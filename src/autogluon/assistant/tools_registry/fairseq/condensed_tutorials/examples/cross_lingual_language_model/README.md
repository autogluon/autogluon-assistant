# Condensed: Cross-Lingual Language Model Pre-training

Summary: This tutorial provides implementation details for training XLM-style models with Masked Language Modeling (MLM) in Fairseq. It covers data preparation using Wikipedia monolingual data, Fairseq pre-processing commands for multiple languages, and the specific training configuration for cross-lingual language models. Key functionalities include handling multilingual data, vocabulary management, and model training with parameters optimized for MLM tasks. The tutorial helps with preprocessing multilingual datasets, configuring Fairseq for cross-lingual training, and implementing memory-efficient training workflows, while noting current limitations regarding perplexity evaluation and downstream fine-tuning.

*This is a condensed version that preserves essential implementation details and context.*

# Cross-Lingual Language Model Pre-training

This guide covers implementation details for training XLM-style models with Masked Language Modeling (MLM) in Fairseq.

## Data Preparation

1. Process monolingual Wikipedia data following the [XLM Github Repository](https://github.com/facebookresearch/XLM#download--preprocess-monolingual-data) instructions
2. Expected file structure:
   - Data in `monolingual_data/processed`
   - Files per language: `train.{lang}`, `valid.{lang}`
   - Vocabulary file: `monolingual_data/processed/vocab_mlm`

## Fairseq Pre-processing

```bash
DATA_DIR=monolingual_data/fairseq_processed
mkdir -p "$DATA_DIR"

for lg in ar de en hi fr
do
  fairseq-preprocess \
  --task cross_lingual_lm \
  --srcdict monolingual_data/processed/vocab_mlm \
  --only-source \
  --trainpref monolingual_data/processed/train \
  --validpref monolingual_data/processed/valid \
  --testpref monolingual_data/processed/test \
  --destdir monolingual_data/fairseq_processed \
  --workers 20 \
  --source-lang $lg

  # Fix output filenames
  for stage in train test valid
    sudo mv "$DATA_DIR/$stage.$lg-None.$lg.bin" "$stage.$lg.bin"
    sudo mv "$DATA_DIR/$stage.$lg-None.$lg.idx" "$stage.$lg.idx"
  done
done
```

## Training the XLM MLM Model

```bash
fairseq-train \
--task cross_lingual_lm monolingual_data/fairseq_processed \
--save-dir checkpoints/mlm \
--max-update 2400000 --save-interval 1 --no-epoch-checkpoints \
--arch xlm_base \
--optimizer adam --lr-scheduler reduce_lr_on_plateau \
--lr-shrink 0.5 --lr 0.0001 --stop-min-lr 1e-09 \
--dropout 0.1 \
--criterion legacy_masked_lm_loss \
--max-tokens 2048 --tokens-per-sample 256 --attention-dropout 0.1 \
--dataset-impl lazy --seed 0 \
--masked-lm-only \
--monolingual-langs 'ar,de,en,hi,fr' --num-segment 5 \
--ddp-backend=legacy_ddp
```

## Important Notes

- **Memory Warning**: Using `tokens_per_sample` > 256 may cause OOM issues
- **Limitations**: 
  - MLM perplexity evaluation workflow is still in progress
  - Downstream task fine-tuning is not currently available