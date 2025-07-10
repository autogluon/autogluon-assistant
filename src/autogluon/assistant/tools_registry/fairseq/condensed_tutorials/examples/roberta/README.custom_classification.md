# Condensed: Finetuning RoBERTa on a custom classification task

Summary: This tutorial demonstrates how to finetune RoBERTa for sentiment classification using the IMDB dataset. It covers the complete workflow: data acquisition, formatting text files into consolidated datasets, BPE encoding with GPT-2 tokenizer, preprocessing with fairseq, and model training with hyperparameter recommendations. Key features include handling classification tasks with pretrained RoBERTa, managing GPU memory constraints through batch size and update frequency adjustments, and performing inference with the finetuned model. This tutorial helps with implementing text classification pipelines, optimizing training configurations, and utilizing the fairseq library for transformer-based NLP tasks.

*This is a condensed version that preserves essential implementation details and context.*

# Finetuning RoBERTa on Custom Classification Tasks

## 1) Data Acquisition
```bash
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar zxvf aclImdb_v1.tar.gz
```

## 2) Data Formatting
Convert individual files into consolidated train/validation files:

```python
import argparse, os, random
from glob import glob

random.seed(0)

def main(args):
    for split in ['train', 'test']:
        samples = []
        for class_label in ['pos', 'neg']:
            fnames = glob(os.path.join(args.datadir, split, class_label) + '/*.txt')
            for fname in fnames:
                with open(fname) as fin:
                    line = fin.readline()
                    samples.append((line, 1 if class_label == 'pos' else 0))
        random.shuffle(samples)
        out_fname = 'train' if split == 'train' else 'dev'
        f1 = open(os.path.join(args.datadir, out_fname + '.input0'), 'w')
        f2 = open(os.path.join(args.datadir, out_fname + '.label'), 'w')
        for sample in samples:
            f1.write(sample[0] + '\n')
            f2.write(str(sample[1]) + '\n')
        f1.close()
        f2.close()
```

## 3) BPE Encoding
```bash
# Download required files
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'

# Encode data
for SPLIT in train dev; do
    python -m examples.roberta.multiprocessing_bpe_encoder \
        --encoder-json encoder.json \
        --vocab-bpe vocab.bpe \
        --inputs "aclImdb/$SPLIT.input0" \
        --outputs "aclImdb/$SPLIT.input0.bpe" \
        --workers 60 \
        --keep-empty
done
```

## 4) Data Preprocessing
```bash
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'  

fairseq-preprocess \
    --only-source \
    --trainpref "aclImdb/train.input0.bpe" \
    --validpref "aclImdb/dev.input0.bpe" \
    --destdir "IMDB-bin/input0" \
    --workers 60 \
    --srcdict dict.txt

fairseq-preprocess \
    --only-source \
    --trainpref "aclImdb/train.label" \
    --validpref "aclImdb/dev.label" \
    --destdir "IMDB-bin/label" \
    --workers 60
```

## 5) Model Training
```bash
TOTAL_NUM_UPDATES=7812  # 10 epochs through IMDB for bsz 32
WARMUP_UPDATES=469      # 6 percent of the number of updates
LR=1e-05                # Peak LR for polynomial LR scheduler
HEAD_NAME=imdb_head     # Classification head name
NUM_CLASSES=2           # Number of classes
MAX_SENTENCES=8         # Batch size
ROBERTA_PATH=/path/to/roberta.large/model.pt

fairseq-train IMDB-bin/ \
    --restore-file $ROBERTA_PATH \
    --max-positions 512 \
    --batch-size $MAX_SENTENCES \
    --max-tokens 4400 \
    --task sentence_prediction \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --init-token 0 --separator-token 2 \
    --arch roberta_large \
    --criterion sentence_prediction \
    --classification-head-name $HEAD_NAME \
    --num-classes $NUM_CLASSES \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
    --max-epoch 10 \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --shorten-method "truncate" \
    --find-unused-parameters \
    --update-freq 4
```

**Important Notes:**
- Expected validation accuracy: ~96.5% after 10 epochs
- If you run out of GPU memory, decrease `--batch-size` and increase `--update-freq` accordingly
- Effective batch size = `batch-size` × `update-freq` (32 in this example)

## 6) Model Inference
```python
from fairseq.models.roberta import RobertaModel
roberta = RobertaModel.from_pretrained(
    'checkpoints',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='IMDB-bin'
)
roberta.eval()  # disable dropout

# Define label function
label_fn = lambda label: roberta.task.label_dictionary.string(
    [label + roberta.task.label_dictionary.nspecial]
)

# Make predictions
tokens = roberta.encode('Best movie this year')
pred = label_fn(roberta.predict('imdb_head', tokens).argmax().item())
assert pred == '1'  # positive

tokens = roberta.encode('Worst movie ever')
pred = label_fn(roberta.predict('imdb_head', tokens).argmax().item())
assert pred == '0'  # negative
```