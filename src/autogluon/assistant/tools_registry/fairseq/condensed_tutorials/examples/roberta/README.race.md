# Condensed: Finetuning RoBERTa on RACE tasks

Summary: This tutorial demonstrates how to finetune RoBERTa models on the RACE (Reading Comprehension from Examinations) multiple-choice task using Fairseq. It covers data preprocessing with specific scripts, model fine-tuning with detailed hyperparameters for handling long contexts, and evaluation procedures. Key implementation techniques include gradient accumulation to simulate larger batch sizes on limited GPU memory, handling long context windows up to 512 tokens, and proper configuration for multiple-choice tasks using sentence ranking. The tutorial provides complete command-line instructions for training and evaluation, with specific guidance on memory optimization for large language models on tasks requiring extensive context processing.

*This is a condensed version that preserves essential implementation details and context.*

# Finetuning RoBERTa on RACE Tasks

## Data Preparation
1. Download RACE data from http://www.cs.cmu.edu/~glai1/data/race/
2. Preprocess the data:
```bash
python ./examples/roberta/preprocess_RACE.py --input-dir <input-dir> --output-dir <extracted-data-dir>
./examples/roberta/preprocess_RACE.sh <extracted-data-dir> <output-dir>
```

## Fine-tuning
```bash
MAX_EPOCH=5           # Training epochs
LR=1e-05              # Peak learning rate
NUM_CLASSES=4
MAX_SENTENCES=1       # Batch size per GPU
UPDATE_FREQ=8         # Gradient accumulation (simulates 8 GPUs)
DATA_DIR=/path/to/race-output-dir
ROBERTA_PATH=/path/to/roberta/model.pt

CUDA_VISIBLE_DEVICES=0,1 fairseq-train $DATA_DIR --ddp-backend=legacy_ddp \
    --restore-file $ROBERTA_PATH \
    --reset-optimizer --reset-dataloader --reset-meters \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --task sentence_ranking \
    --num-classes $NUM_CLASSES \
    --init-token 0 --separator-token 2 \
    --max-option-length 128 \
    --max-positions 512 \
    --shorten-method "truncate" \
    --arch roberta_large \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --criterion sentence_ranking \
    --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler fixed --lr $LR \
    --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
    --batch-size $MAX_SENTENCES \
    --required-batch-size-multiple 1 \
    --update-freq $UPDATE_FREQ \
    --max-epoch $MAX_EPOCH
```

## Important Implementation Notes
- Use smaller batch size with increased `update-freq` for long contexts in RACE
- Tested on Nvidia V100 (32GB) - adjust `--update-freq` and `--batch-size` based on available GPU memory
- Hyperparameters are from a fixed search space - better metrics may be possible with wider search

## Evaluation
```bash
DATA_DIR=/path/to/race-output-dir
MODEL_PATH=/path/to/checkpoint_best.pt
PREDS_OUT=preds.tsv
TEST_SPLIT=test       # test (Middle) or test1 (High)

fairseq-validate \
    $DATA_DIR \
    --valid-subset $TEST_SPLIT \
    --path $MODEL_PATH \
    --batch-size 1 \
    --task sentence_ranking \
    --criterion sentence_ranking \
    --save-predictions $PREDS_OUT
```