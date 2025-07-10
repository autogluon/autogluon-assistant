# Condensed: [[Back]](..)

Summary: This tutorial demonstrates implementing automatic speech recognition (ASR) on LibriSpeech using Fairseq's Speech-to-Text (S2T) framework. It covers data preparation with sentencepiece tokenization, training transformer-based ASR models with different parameter sizes (30M-268M), and evaluation techniques. Key functionalities include command-line scripts for preprocessing audio data, training with various hyperparameter configurations, checkpoint averaging for improved performance, and inference with beam search. The tutorial provides practical knowledge for building speech recognition systems with WER metrics on standard benchmarks and includes interactive decoding capabilities for real-time transcription of audio files.

*This is a condensed version that preserves essential implementation details and context.*

# S2T Example: Speech Recognition (ASR) on LibriSpeech

## Data Preparation
```bash
# Install required packages
pip install pandas torchaudio sentencepiece

# Preprocess LibriSpeech data
python examples/speech_to_text/prep_librispeech_data.py \
  --output-root ${LS_ROOT} --vocab-type unigram --vocab-size 10000
```

You can download pre-trained vocabulary files from [here](https://dl.fbaipublicfiles.com/fairseq/s2t/librispeech_vocab_unigram10000.zip).

## Training
```bash
fairseq-train ${LS_ROOT} --save-dir ${SAVE_DIR} \
  --config-yaml config.yaml --train-subset train-clean-100,train-clean-360,train-other-500 --valid-subset dev-clean,dev-other \
  --num-workers 4 --max-tokens 40000 --max-update 300000 \
  --task speech_to_text --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --report-accuracy \
  --arch s2t_transformer_s --share-decoder-input-output-embed \
  --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt --warmup-updates 10000 \
  --clip-norm 10.0 --seed 1 --update-freq 8
```

**Model Architecture Options:**
- `s2t_transformer_s`: 31M parameters, lr=2e-3
- `s2t_transformer_m`: 71M parameters, lr=1e-3
- `s2t_transformer_l`: 268M parameters, lr=5e-4

Note: `--update-freq 8` simulates 8 GPUs with 1 GPU. Adjust according to your setup.

## Inference & Evaluation
```bash
# Average last 10 checkpoints
CHECKPOINT_FILENAME=avg_last_10_checkpoint.pt
python scripts/average_checkpoints.py --inputs ${SAVE_DIR} \
  --num-epoch-checkpoints 10 \
  --output "${SAVE_DIR}/${CHECKPOINT_FILENAME}"

# Evaluate on all splits
for SUBSET in dev-clean dev-other test-clean test-other; do
  fairseq-generate ${LS_ROOT} --config-yaml config.yaml --gen-subset ${SUBSET} \
    --task speech_to_text --path ${SAVE_DIR}/${CHECKPOINT_FILENAME} \
    --max-tokens 50000 --beam 5 --scoring wer
done
```

## Interactive Decoding
```bash
fairseq-interactive ${LS_ROOT} --config-yaml config.yaml --task speech_to_text \
  --path ${SAVE_DIR}/${CHECKPOINT_FILENAME} --max-tokens 50000 --beam 5
```
Type WAV/FLAC/OGG audio paths (one per line) at the prompt.

## Results

| Architecture | Params | dev-clean | dev-other | test-clean | test-other |
|---|---|---|---|---|---|
| s2t_transformer_s | 30M | 3.8 | 8.9 | 4.4 | 9.0 |
| s2t_transformer_m | 71M | 3.2 | 8.0 | 3.4 | 7.9 |
| s2t_transformer_l | 268M | 3.0 | 7.5 | 3.2 | 7.5 |

Pre-trained models are available for download in the original documentation.