# Condensed: Fine-tuning BART on CNN-Dailymail summarization task

Summary: This tutorial provides a comprehensive guide for fine-tuning BART on text summarization tasks (CNN-DM or XSum datasets). It covers the complete implementation workflow including data preparation, BPE preprocessing with GPT-2 tokenizer, dataset binarization using fairseq-preprocess, and model fine-tuning with detailed training parameters. The tutorial includes specific code for handling different datasets, hardware requirements (8 V100 GPUs), and inference procedures. Key functionalities include transfer learning with pre-trained BART, hyperparameter optimization for summarization tasks, and dataset-specific configurations for generating summaries with appropriate length constraints.

*This is a condensed version that preserves essential implementation details and context.*

# Fine-tuning BART on CNN-Dailymail Summarization Task

## Data Preparation

1. Download and preprocess CNN-DM or XSum datasets:
   - CNN-DM: Follow instructions at [GitHub repo](https://github.com/abisee/cnn-dailymail)
   - XSum: Follow instructions at [XSum repo](https://github.com/EdinburghNLP/XSum)
   - **Important**: Keep raw dataset with no tokenization or BPE

## BPE Preprocessing

```bash
# Download required files
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'

# Process splits
TASK=cnn_dm
for SPLIT in train val
do
  for LANG in source target
  do
    python -m examples.roberta.multiprocessing_bpe_encoder \
    --encoder-json encoder.json \
    --vocab-bpe vocab.bpe \
    --inputs "$TASK/$SPLIT.$LANG" \
    --outputs "$TASK/$SPLIT.bpe.$LANG" \
    --workers 60 \
    --keep-empty;
  done
done
```

## Binarize Dataset

```bash
fairseq-preprocess \
  --source-lang "source" \
  --target-lang "target" \
  --trainpref "${TASK}/train.bpe" \
  --validpref "${TASK}/val.bpe" \
  --destdir "${TASK}-bin/" \
  --workers 60 \
  --srcdict dict.txt \
  --tgtdict dict.txt;
```

## Fine-tuning

```bash
# Key parameters
TOTAL_NUM_UPDATES=20000  # Use 15000 for XSum
WARMUP_UPDATES=500      
LR=3e-05
MAX_TOKENS=2048
UPDATE_FREQ=4           # Use 2 for XSum
BART_PATH=/path/to/bart/model.pt

# Training command
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 fairseq-train cnn_dm-bin \
    --restore-file $BART_PATH \
    --max-tokens $MAX_TOKENS \
    --task translation \
    --source-lang source --target-lang target \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --arch bart_large \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16 --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters;
```

**Hardware Requirements**: 
- 1 node with 8 32GB V100 GPUs
- Expected training time: ~5 hours
- Can reduce training time with distributed training on 4 nodes and `--update-freq 1`

## Inference

```bash
# Copy dictionary file
cp data-bin/cnn_dm/dict.source.txt checkpoints/

# For CNN-DM
python examples/bart/summarize.py \
  --model-dir checkpoints \
  --model-file checkpoint_best.pt \
  --src cnn_dm/test.source \
  --out cnn_dm/test.hypo

# For XSum (uses beam=6, lenpen=1.0, max_len_b=60, min_len=10)
python examples/bart/summarize.py \
  --model-dir checkpoints \
  --model-file checkpoint_best.pt \
  --src cnn_dm/test.source \
  --out cnn_dm/test.hypo \
  --xsum-kwargs
```