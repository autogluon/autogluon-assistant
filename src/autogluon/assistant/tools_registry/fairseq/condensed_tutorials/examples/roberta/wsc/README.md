# Condensed: Finetuning RoBERTa on Winograd Schema Challenge (WSC) data

Summary: This tutorial demonstrates how to finetune RoBERTa on the Winograd Schema Challenge (WSC) for pronoun disambiguation. It covers implementation of a cross-entropy loss approach that outperforms margin ranking loss, candidate mining using spaCy, and complete training configuration with hyperparameters. The code shows how to set up data, train models with multiple GPUs (with single GPU adaptation), and evaluate model performance. Key functionalities include customized loss functions, candidate mining techniques, and model evaluation methods. The tutorial also provides configuration for the WinoGrande dataset variant. This knowledge helps with implementing pronoun disambiguation systems and fine-tuning transformer models for coreference resolution tasks.

*This is a condensed version that preserves essential implementation details and context.*

# Finetuning RoBERTa on Winograd Schema Challenge (WSC)

## Key Implementation Details

- **Loss Function**: Uses cross entropy loss over log-probabilities for query and mined candidates instead of margin ranking loss described in original paper
- **Candidate Mining**: Candidates are mined using spaCy from each input sentence in isolation (pointwise approach)
- **Performance**: Best model achieved 92.3% development set accuracy vs ~90% for margin loss
- **High Variance Warning**: Results show high variance; official submission used ensemble of 7 models from ~100 runs

## Setup and Data Preparation

```bash
wget https://dl.fbaipublicfiles.com/glue/superglue/data/v2/WSC.zip
unzip WSC.zip
wget -O WSC/dict.txt https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt
```

## Training Configuration

```bash
TOTAL_NUM_UPDATES=2000  # Total training steps
WARMUP_UPDATES=250      # Linear LR warmup steps
LR=2e-05                # Peak learning rate
MAX_SENTENCES=16        # Batch size per GPU
SEED=1                  # Random seed
ROBERTA_PATH=/path/to/roberta/model.pt
FAIRSEQ_USER_DIR=${FAIRSEQ_PATH}/examples/roberta/wsc

CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-train WSC/ \
    --restore-file $ROBERTA_PATH \
    --reset-optimizer --reset-dataloader --reset-meters \
    --no-epoch-checkpoints --no-last-checkpoints --no-save-optimizer-state \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --valid-subset val \
    --fp16 --ddp-backend legacy_ddp \
    --user-dir $FAIRSEQ_USER_DIR \
    --task wsc --criterion wsc --wsc-cross-entropy \
    --arch roberta_large --bpe gpt2 --max-positions 512 \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-06 \
    --lr-scheduler polynomial_decay --lr $LR \
    --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_NUM_UPDATES \
    --batch-size $MAX_SENTENCES \
    --max-update $TOTAL_NUM_UPDATES \
    --log-format simple --log-interval 100 \
    --seed $SEED
```

**Note**: For single GPU, add `--update-freq=4` to achieve same results.

## Evaluation

```python
from fairseq.models.roberta import RobertaModel
from examples.roberta.wsc import wsc_utils  # also loads WSC task and criterion
roberta = RobertaModel.from_pretrained('checkpoints', 'checkpoint_best.pt', 'WSC/')
roberta.cuda()
nsamples, ncorrect = 0, 0
for sentence, label in wsc_utils.jsonl_iterator('WSC/val.jsonl', eval=True):
    pred = roberta.disambiguate_pronoun(sentence)
    nsamples += 1
    if pred == label:
        ncorrect += 1
print('Accuracy: ' + str(ncorrect / float(nsamples)))
# Accuracy: 0.9230769230769231
```

## WinoGrande Dataset Training

For datasets with exactly two candidates (one correct):

```bash
TOTAL_NUM_UPDATES=23750 # Total training steps
WARMUP_UPDATES=2375     # Linear LR warmup steps
LR=1e-05                # Peak learning rate
MAX_SENTENCES=32        # Batch size per GPU

# Key parameters for WinoGrande
--task winogrande --criterion winogrande \
--wsc-margin-alpha 5.0 --wsc-margin-beta 0.4 \
```