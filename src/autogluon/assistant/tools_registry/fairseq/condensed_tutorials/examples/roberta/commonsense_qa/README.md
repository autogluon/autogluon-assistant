# Condensed: Finetuning RoBERTa on Commonsense QA

Summary: This tutorial demonstrates how to finetune RoBERTa for multiple-choice commonsense question answering using the CommonsenseQA dataset. It covers a specific approach where each question-answer pair is formatted with special prefixes and processed through a sentence ranking task. Key implementation details include data preparation, model finetuning with hyperparameters (learning rate, batch size, warmup steps), and evaluation code. The tutorial provides complete code for training with fairseq (including command-line arguments) and evaluation using PyTorch, achieving 78.5% accuracy. This is valuable for implementing multiple-choice QA systems or adapting pretrained language models for classification tasks.

*This is a condensed version that preserves essential implementation details and context.*

# Finetuning RoBERTa on Commonsense QA

## Approach
- For each question, construct five inputs (one per answer choice)
- Format: concatenate question and candidate answer with special prefixes
- Pass "[CLS]" representations through a fully-connected layer
- Train with standard cross-entropy loss

## Input Format
```
<s> Q: Where would I not want a fox? </s> A: hen house </s>
```

## Implementation Steps

### 1) Download Data
```bash
bash examples/roberta/commonsense_qa/download_cqa_data.sh
```

### 2) Finetune
```bash
MAX_UPDATES=3000      # Training steps
WARMUP_UPDATES=150    # LR warmup steps
LR=1e-05              # Peak learning rate
MAX_SENTENCES=16      # Batch size
SEED=1                # Random seed
ROBERTA_PATH=/path/to/roberta/model.pt
DATA_DIR=data/CommonsenseQA
FAIRSEQ_PATH=/path/to/fairseq
FAIRSEQ_USER_DIR=${FAIRSEQ_PATH}/examples/roberta/commonsense_qa

CUDA_VISIBLE_DEVICES=0 fairseq-train --fp16 --ddp-backend=legacy_ddp \
    $DATA_DIR \
    --user-dir $FAIRSEQ_USER_DIR \
    --restore-file $ROBERTA_PATH \
    --reset-optimizer --reset-dataloader --reset-meters \
    --no-epoch-checkpoints --no-last-checkpoints --no-save-optimizer-state \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --task commonsense_qa --init-token 0 --bpe gpt2 \
    --arch roberta_large --max-positions 512 \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --criterion sentence_ranking --num-classes 5 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-06 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR \
    --warmup-updates $WARMUP_UPDATES --total-num-update $MAX_UPDATES \
    --batch-size $MAX_SENTENCES \
    --max-update $MAX_UPDATES \
    --log-format simple --log-interval 25 \
    --seed $SEED
```

**Note:** For GPUs with less than 32GB RAM, decrease `--batch-size` and increase `--update-freq` accordingly.

### 3) Evaluate
```python
import json
import torch
from fairseq.models.roberta import RobertaModel
from examples.roberta import commonsense_qa  # load the Commonsense QA task

roberta = RobertaModel.from_pretrained('checkpoints', 'checkpoint_best.pt', 'data/CommonsenseQA')
roberta.eval()  # disable dropout
roberta.cuda()  # use GPU (optional)

nsamples, ncorrect = 0, 0
with open('data/CommonsenseQA/valid.jsonl') as h:
    for line in h:
        example = json.loads(line)
        scores = []
        for choice in example['question']['choices']:
            input = roberta.encode(
                'Q: ' + example['question']['stem'],
                'A: ' + choice['text'],
                no_separator=True
            )
            score = roberta.predict('sentence_classification_head', input, return_logits=True)
            scores.append(score)
        pred = torch.cat(scores).argmax()
        answer = ord(example['answerKey']) - ord('A')
        nsamples += 1
        if pred == answer:
            ncorrect += 1

print('Accuracy: ' + str(ncorrect / float(nsamples)))
# Accuracy: 0.7846027846027847
```

**Note:** For faster inference, use batched prediction as described in the RoBERTa documentation.