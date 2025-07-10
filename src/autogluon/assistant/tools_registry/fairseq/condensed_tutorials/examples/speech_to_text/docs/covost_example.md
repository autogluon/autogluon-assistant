# Condensed: [[Back]](..)

Summary: This tutorial demonstrates implementing speech-to-text (S2T) translation using the CoVoST dataset with Fairseq. It covers data preparation, ASR and ST model training, inference, and evaluation workflows. Key technical aspects include transformer and conformer architectures, positional encoding options (abs, rope, rel_pos), checkpoint averaging, and transfer learning via pre-trained ASR encoders. The tutorial provides complete command-line examples for training models with various hyperparameters, conducting inference, and evaluating performance using WER and BLEU metrics. Developers can use this to implement multilingual speech translation systems with state-of-the-art architectures and leverage pre-trained models available for multiple language pairs.

*This is a condensed version that preserves essential implementation details and context.*

# S2T Example: ST on CoVoST

This tutorial replicates experiments from [CoVoST 2 and Massively Multilingual Speech-to-Text Translation](https://arxiv.org/abs/2007.10310).

## Data Preparation

1. Download and unpack Common Voice v4 to `${COVOST_ROOT}/${SOURCE_LANG_ID}`
2. Process the data:

```bash
# Install dependencies
pip install pandas torchaudio sentencepiece

# For En ASR
python examples/speech_to_text/prep_covost_data.py \
  --data-root ${COVOST_ROOT} --vocab-type char --src-lang en

# For ST
python examples/speech_to_text/prep_covost_data.py \
  --data-root ${COVOST_ROOT} --vocab-type char \
  --src-lang fr --tgt-lang en
```

Pre-trained vocabulary files are available for download for various language pairs.

## ASR Training

```bash
fairseq-train ${COVOST_ROOT}/en \
  --config-yaml config_asr_en.yaml --train-subset train_asr_en --valid-subset dev_asr_en \
  --save-dir ${ASR_SAVE_DIR} --num-workers 4 --max-tokens 50000 --max-update 60000 \
  --task speech_to_text --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --report-accuracy --arch s2t_transformer_s --dropout 0.15 --optimizer adam --lr 2e-3 \
  --lr-scheduler inverse_sqrt --warmup-updates 10000 --clip-norm 10.0 --seed 1 --update-freq 8 \
  --attn-type None --pos-enc-type ${POS_ENC_TYPE}
```

**Key parameters:**
- `--pos-enc-type`: Use `abs`, `rope`, or `rel_pos` for positional encoding
- `--attn-type espnet`: Required to use conformer encoder
- `--update-freq 8`: Simulates 8 GPUs with 1 GPU

## ASR Inference & Evaluation

```bash
# Average last 10 checkpoints
CHECKPOINT_FILENAME=avg_last_10_checkpoint.pt
python scripts/average_checkpoints.py \
  --inputs ${ASR_SAVE_DIR} --num-epoch-checkpoints 10 \
  --output "${ASR_SAVE_DIR}/${CHECKPOINT_FILENAME}"

# Generate and evaluate
fairseq-generate ${COVOST_ROOT}/en \
  --config-yaml config_asr_en.yaml --gen-subset test_asr_en --task speech_to_text \
  --path ${ASR_SAVE_DIR}/${CHECKPOINT_FILENAME} --max-tokens 50000 --beam 5 \
  --scoring wer --wer-tokenizer 13a --wer-lowercase --wer-remove-punct
```

## ST Training

Example for Fr-En:

```bash
fairseq-train ${COVOST_ROOT}/fr \
  --config-yaml config_st_fr_en.yaml --train-subset train_st_fr_en --valid-subset dev_st_fr_en \
  --save-dir ${ST_SAVE_DIR} --num-workers 4 --max-update 30000 --max-tokens 40000 \  # --max-tokens 50000 for en-*
  --task speech_to_text --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --report-accuracy \
  --arch s2t_transformer_s --encoder-freezing-updates 1000 --optimizer adam --lr 2e-3 \
  --lr-scheduler inverse_sqrt --warmup-updates 10000 --clip-norm 10.0 --seed 1 --update-freq 8 \
  --attn-type None --pos-enc-type ${POS_ENC_TYPE} \
  --load-pretrained-encoder-from ${ASR_SAVE_DIR}/${CHECKPOINT_FILENAME}
```

**Important options:**
- Optional ASR pre-training: `--load-pretrained-encoder-from <ASR checkpoint path>`
- Use `--max-tokens 50000` for en-* language pairs

## ST Inference & Evaluation

```bash
# Average last 10 checkpoints
CHECKPOINT_FILENAME=avg_last_10_checkpoint.pt
python scripts/average_checkpoints.py \
  --inputs ${ST_SAVE_DIR} --num-epoch-checkpoints 10 \
  --output "${ST_SAVE_DIR}/${CHECKPOINT_FILENAME}"

# Generate and evaluate
fairseq-generate ${COVOST_ROOT}/fr \
  --config-yaml config_st_fr_en.yaml --gen-subset test_st_fr_en --task speech_to_text \
  --path ${ST_SAVE_DIR}/${CHECKPOINT_FILENAME} \
  --max-tokens 50000 --beam 5 --scoring sacrebleu
```

## Interactive Decoding

```bash
fairseq-interactive ${COVOST_ROOT}/fr --config-yaml config_st_fr_en.yaml \
  --task speech_to_text --path ${SAVE_DIR}/${CHECKPOINT_FILENAME} \
  --max-tokens 50000 --beam 5
```

## Results Summary

### ASR Results
| Architecture | Pos Encoding | Params | WER (En) |
|---|---|---|---|
| s2t_transformer_s | - | 31M | 25.6 |
| s2t_conformer | rel_pos | 42.9M | 23.18 |
| s2t_conformer | rope | 42.1M | 23.8 |
| s2t_conformer | abs | 42.1M | 23.8 |

### ST Results
Best BLEU scores:
- Fr-En: 28.32 (conformer with rel_pos)
- De-En: 18.22 (conformer with rel_pos + ASR PT)
- Es-En: 25.98 (conformer with rel_pos)
- En-De: 20.37 (conformer with rel_pos)
- En-Ca: 25.92 (conformer with rel_pos + ASR PT)

Pre-trained models are available for download for all language pairs and architectures.