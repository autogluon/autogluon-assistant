# Condensed: An example of English to Japaneses Simultaneous Translation System

Summary: This tutorial demonstrates implementing a transformer wait-k model for English-to-Japanese simultaneous translation using fairseq. It covers data preprocessing with sentencepiece tokenization, training a transformer monotonic model with specific wait-k parameters, and evaluation using SimulEval. Key functionalities include configuring wait-k lagging for simultaneous translation, handling Japanese-specific tokenization with MeCab, and measuring translation quality (BLEU) alongside latency metrics (AL, AP, DAL). The tutorial provides complete command-line implementations for preprocessing, training, and evaluation, making it valuable for developers building real-time translation systems with specific latency-quality tradeoffs.

*This is a condensed version that preserves essential implementation details and context.*

# English to Japanese Simultaneous Translation System

This tutorial demonstrates how to train and evaluate a transformer *wait-k* model for English to Japanese simultaneous text-to-text translation.

## Data Preparation

```bash
fairseq-preprocess \
    --source-lang en --target-lang ja \
    --trainpref ${DATA_DIR}/train \
    --validpref ${DATA_DIR}/dev \
    --testpref ${DATA_DIR}/test \
    --destdir ${WMT20_ENJA_DATA_BIN} \
    --nwordstgt 32000 --nwordssrc 32000 \
    --workers 20
```

Key points:
- Uses WMT20 news translation task data (7,815,391 sentence pairs)
- Tokenized with sentencepiece (vocab size: 32000)
- Sentences longer than 200 words are filtered out

## Model Training

```bash
fairseq-train ${WMT20_ENJA_DATA_BIN} \
    --save-dir ${SAVEDIR} \
    --simul-type waitk \
    --waitk-lagging 10 \
    --max-epoch 70 \
    --arch transformer_monotonic_vaswani_wmt_en_de_big \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt \
    --warmup-init-lr 1e-07 \
    --warmup-updates 4000 \
    --lr 0.0005 \
    --stop-min-lr 1e-09 \
    --clip-norm 10.0 \
    --dropout 0.3 \
    --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --max-tokens 3584
```

Note: This command is for 8 GPUs. For single GPU, use `--update-freq 8`.

## Inference & Evaluation

1. Install SimulEval:
```bash
git clone https://github.com/facebookresearch/SimulEval.git
cd SimulEval
pip install -e .
```

2. Run evaluation:
```bash
simuleval \
    --source ${SRC_FILE} \
    --target ${TGT_FILE} \
    --data-bin ${WMT20_ENJA_DATA_BIN} \
    --sacrebleu-tokenizer ja-mecab \
    --eval-latency-unit char \
    --no-space \
    --src-splitter-type sentencepiecemodel \
    --src-splitter-path ${SRC_SPM_PATH} \
    --agent ${FAIRSEQ}/examples/simultaneous_translation/agents/simul_trans_text_agent_enja.py \
    --model-path ${SAVE_DIR}/${CHECKPOINT_FILENAME} \
    --output ${OUTPUT} \
    --scores
```

Important configurations:
- `--eval-latency-unit char`: Evaluates latency by characters on target side
- `--sacrebleu-tokenizer ja-mecab`: Uses MeCab tokenizer for Japanese
- `--no-space`: No spaces when merging predicted words

Pre-trained resources:
- Data directory: [wmt20_enja_medium_databin.tgz](https://dl.fbaipublicfiles.com/simultaneous_translation/wmt20_enja_medium_databin.tgz)
- Pre-trained wait-k=10 model: [wmt20_enja_medium_wait10_ckpt.pt](https://dl.fbaipublicfiles.com/simultaneous_translation/wmt20_enja_medium_wait10_ckpt.pt)

Expected output:
```json
{
    "Quality": {
        "BLEU": 11.442253287568398
    },
    "Latency": {
        "AL": 8.6587861866951,
        "AP": 0.7863304776251316,
        "DAL": 9.477850951194764
    }
}
```