# Condensed: [[Back]](..)

Summary: This tutorial demonstrates speech translation implementation using the MuST-C dataset of TED talks. It covers techniques for building both single-language and multilingual speech-to-text systems, including data preparation with custom vocabularies, ASR pre-training for encoder initialization, and multilingual modeling with language ID tokens. Key functionalities include training pipelines for ASR and ST models, checkpoint averaging for improved performance, inference procedures with beam search, and evaluation using WER and BLEU metrics. The tutorial provides complete code examples for data preprocessing, model training with transformer architectures, and evaluation workflows for both bilingual and multilingual speech translation systems.

*This is a condensed version that preserves essential implementation details and context.*

# Speech Translation (ST) on MuST-C

This tutorial demonstrates how to implement speech translation using the MuST-C dataset, which contains 8-language translations of English TED talks.

## Data Preparation

```bash
# Install required packages
pip install pandas torchaudio soundfile sentencepiece

# Generate TSV manifests, features, vocabulary for each language
python examples/speech_to_text/prep_mustc_data.py \
  --data-root ${MUSTC_ROOT} --task asr \
  --vocab-type unigram --vocab-size 5000
python examples/speech_to_text/prep_mustc_data.py \
  --data-root ${MUSTC_ROOT} --task st \
  --vocab-type unigram --vocab-size 8000

# Add vocabulary and configuration for joint data
python examples/speech_to_text/prep_mustc_data.py \
  --data-root ${MUSTC_ROOT} --task asr --joint \
  --vocab-type unigram --vocab-size 10000
python examples/speech_to_text/prep_mustc_data.py \
  --data-root ${MUSTC_ROOT} --task st --joint \
  --vocab-type unigram --vocab-size 10000
```

Pre-trained vocabulary files are available for download for both ASR and ST tasks.

## ASR Training

```bash
# Single language (En-De example)
fairseq-train ${MUSTC_ROOT}/en-de \
  --config-yaml config_asr.yaml --train-subset train_asr --valid-subset dev_asr \
  --save-dir ${ASR_SAVE_DIR} --num-workers 4 --max-tokens 40000 --max-update 100000 \
  --task speech_to_text --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --arch s2t_transformer_s --optimizer adam --lr 1e-3 --lr-scheduler inverse_sqrt \
  --warmup-updates 10000 --clip-norm 10.0 --seed 1 --update-freq 8

# Joint model (all 8 directions)
fairseq-train ${MUSTC_ROOT} \
  --config-yaml config_asr.yaml \
  --train-subset train_de_asr,train_nl_asr,train_es_asr,train_fr_asr,train_it_asr,train_pt_asr,train_ro_asr,train_ru_asr \
  --valid-subset dev_de_asr,dev_nl_asr,dev_es_asr,dev_fr_asr,dev_it_asr,dev_pt_asr,dev_ro_asr,dev_ru_asr \
  --save-dir ${JOINT_ASR_SAVE_DIR} --num-workers 4 --max-tokens 40000 --max-update 100000 \
  --task speech_to_text --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --arch s2t_transformer_s --optimizer adam --lr 1e-3 --lr-scheduler inverse_sqrt \
  --warmup-updates 10000 --clip-norm 10.0 --seed 1 --update-freq 8
```

**Note:** `--update-freq 8` simulates 8 GPUs with 1 GPU. Adjust according to your hardware.

## ASR Inference & Evaluation

```bash
# Average last 10 checkpoints
CHECKPOINT_FILENAME=avg_last_10_checkpoint.pt
python scripts/average_checkpoints.py \
  --inputs ${ASR_SAVE_DIR} --num-epoch-checkpoints 10 \
  --output "${ASR_SAVE_DIR}/${CHECKPOINT_FILENAME}"

# Evaluate
fairseq-generate ${MUSTC_ROOT}/en-de \
  --config-yaml config_asr.yaml --gen-subset tst-COMMON_asr --task speech_to_text \
  --path ${ASR_SAVE_DIR}/${CHECKPOINT_FILENAME} --max-tokens 50000 --beam 5 \
  --scoring wer --wer-tokenizer 13a --wer-lowercase --wer-remove-punct
```

## ST Training

```bash
# Single language (En-De example)
fairseq-train ${MUSTC_ROOT}/en-de \
  --config-yaml config_st.yaml --train-subset train_st --valid-subset dev_st \
  --save-dir ${ST_SAVE_DIR} --num-workers 4 --max-tokens 40000 --max-update 100000 \
  --task speech_to_text --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --arch s2t_transformer_s --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt \
  --warmup-updates 10000 --clip-norm 10.0 --seed 1 --update-freq 8 \
  --load-pretrained-encoder-from ${ASR_SAVE_DIR}/${CHECKPOINT_FILENAME}

# Multilingual model (all 8 directions)
fairseq-train ${MUSTC_ROOT} \
  --config-yaml config_st.yaml \
  --train-subset train_de_st,train_nl_st,train_es_st,train_fr_st,train_it_st,train_pt_st,train_ro_st,train_ru_st \
  --valid-subset dev_de_st,dev_nl_st,dev_es_st,dev_fr_st,dev_it_st,dev_pt_st,dev_ro_st,dev_ru_st \
  --save-dir ${MULTILINGUAL_ST_SAVE_DIR} --num-workers 4 --max-tokens 40000 --max-update 100000 \
  --task speech_to_text --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --arch s2t_transformer_s --ignore-prefix-size 1 --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt \
  --warmup-updates 10000 --clip-norm 10.0 --seed 1 --update-freq 8 \
  --load-pretrained-encoder-from ${JOINT_ASR_SAVE_DIR}/${CHECKPOINT_FILENAME}
```

**Important implementation details:**
- ST encoder is pre-trained by ASR for faster training and better performance
- For multilingual models, target language ID token is prepended as target BOS
- `--ignore-prefix-size 1` excludes the language ID token from training loss

## ST Inference & Evaluation

```bash
# Average last 10 checkpoints
CHECKPOINT_FILENAME=avg_last_10_checkpoint.pt
python scripts/average_checkpoints.py \
  --inputs ${ST_SAVE_DIR} --num-epoch-checkpoints 10 \
  --output "${ST_SAVE_DIR}/${CHECKPOINT_FILENAME}"

# Evaluate
fairseq-generate ${MUSTC_ROOT}/en-de \
  --config-yaml config_st.yaml --gen-subset tst-COMMON_st --task speech_to_text \
  --path ${ST_SAVE_DIR}/${CHECKPOINT_FILENAME} \
  --max-tokens 50000 --beam 5 --scoring sacrebleu

# For multilingual models, force decoding from target language ID token
fairseq-generate ${MUSTC_ROOT} \
  --config-yaml config_st.yaml --gen-subset tst-COMMON_${LANG}_st --task speech_to_text \
  --prefix-size 1 --path ${MULTILINGUAL_ST_SAVE_DIR}/${CHECKPOINT_FILENAME} \
  --max-tokens 50000 --beam 5 --scoring sacrebleu
```

## Performance Results

**ASR Results (WER)**
- Single language models (s2t_transformer_s, 31M params): 17.2-19.1 WER
- Joint model (s2t_transformer_m, 76M params): 16.7-17.4 WER

**ST Results (BLEU)**
- Bilingual models (s2t_transformer_s, 31M params): 15.3-32.9 BLEU
- Multilingual model (s2t_transformer_m, 76M params): 16.0-34.9 BLEU

Pre-trained models are available for download for all language pairs.