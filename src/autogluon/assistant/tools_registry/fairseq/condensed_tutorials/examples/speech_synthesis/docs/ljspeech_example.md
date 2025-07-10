# Condensed: [[Back]](..)

Summary: This tutorial demonstrates implementing text-to-speech (TTS) models on the LJSpeech dataset using fairseq. It covers data preparation (audio manifest creation, feature extraction), training two architectures (Transformer and FastSpeech2), inference with checkpoint averaging and Griffin-Lim vocoding, and comprehensive evaluation metrics (WER/CER, MCD/MSD, F0 metrics). Key functionalities include handling IPA vocabulary, G2P conversion, FastSpeech2 auxiliary targets (frame durations via force-alignment or pseudo-text units), and various optimization techniques. The tutorial provides complete command-line examples for each step and includes pre-trained model downloads.

*This is a condensed version that preserves essential implementation details and context.*

# LJSpeech

## Data Preparation

1. Download data and create manifests:
```bash
python -m examples.speech_synthesis.preprocessing.get_ljspeech_audio_manifest \
  --output-data-root ${AUDIO_DATA_ROOT} \
  --output-manifest-root ${AUDIO_MANIFEST_ROOT}
```

2. Extract features and create configuration:
```bash
python -m examples.speech_synthesis.preprocessing.get_feature_manifest \
  --audio-manifest-root ${AUDIO_MANIFEST_ROOT} \
  --output-root ${FEATURE_MANIFEST_ROOT} \
  --ipa-vocab --use-g2p
```

3. For FastSpeech 2, add auxiliary targets:
   - Add `--add-fastspeech-targets` flag
   - Specify frame durations using either:
     - `--textgrid-zip ${TEXT_GRID_ZIP_PATH}` for force-alignment
     - `--id-to-units-tsv ${ID_TO_UNIT_TSV}` for pseudo-text units
   - Pre-computed resources available:
     - [Force-alignment](https://dl.fbaipublicfiles.com/fairseq/s2/ljspeech_mfa.zip)
     - [Pseudo-text units](s3://dl.fbaipublicfiles.com/fairseq/s2/ljspeech_hubert.tsv)

## Training

### Transformer
```bash
fairseq-train ${FEATURE_MANIFEST_ROOT} --save-dir ${SAVE_DIR} \
  --config-yaml config.yaml --train-subset train --valid-subset dev \
  --num-workers 4 --max-tokens 30000 --max-update 200000 \
  --task text_to_speech --criterion tacotron2 --arch tts_transformer \
  --clip-norm 5.0 --n-frames-per-step 4 --bce-pos-weight 5.0 \
  --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 \
  --encoder-normalize-before --decoder-normalize-before \
  --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
  --seed 1 --update-freq 8 --eval-inference --best-checkpoint-metric mcd_loss
```

### FastSpeech2
```bash
fairseq-train ${FEATURE_MANIFEST_ROOT} --save-dir ${SAVE_DIR} \
  --config-yaml config.yaml --train-subset train --valid-subset dev \
  --num-workers 4 --max-sentences 6 --max-update 200000 \
  --task text_to_speech --criterion fastspeech2 --arch fastspeech2 \
  --clip-norm 5.0 --n-frames-per-step 1 \
  --dropout 0.1 --attention-dropout 0.1 \
  --optimizer adam --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
  --seed 1 --update-freq 8 --eval-inference --best-checkpoint-metric mcd_loss
```

## Inference

Average checkpoints and generate waveforms:
```bash
SPLIT=test
CHECKPOINT_NAME=avg_last_5
CHECKPOINT_PATH=${SAVE_DIR}/checkpoint_${CHECKPOINT_NAME}.pt

# Average last 5 checkpoints
python scripts/average_checkpoints.py --inputs ${SAVE_DIR} \
  --num-epoch-checkpoints 5 \
  --output ${CHECKPOINT_PATH}

# Generate waveforms using Griffin-Lim
python -m examples.speech_synthesis.generate_waveform ${FEATURE_MANIFEST_ROOT} \
  --config-yaml config.yaml --gen-subset ${SPLIT} --task text_to_speech \
  --path ${CHECKPOINT_PATH} --max-tokens 50000 --spec-bwd-max-iter 32 \
  --dump-waveforms
```

## Automatic Evaluation

1. Generate evaluation manifest:
```bash
python -m examples.speech_synthesis.evaluation.get_eval_manifest \
  --generation-root ${SAVE_DIR}/generate-${CHECKPOINT_NAME}-${SPLIT} \
  --audio-manifest ${AUDIO_MANIFEST_ROOT}/${SPLIT}.audio.tsv \
  --output-path ${EVAL_OUTPUT_ROOT}/eval.tsv \
  --vocoder griffin_lim --sample-rate 22050 --audio-format flac \
  --use-resynthesized-target
```

2. Evaluation metrics:

   - WER/CER (using wav2vec 2.0):
   ```bash
   python -m examples.speech_synthesis.evaluation.eval_asr \
     --audio-header syn --text-header text --err-unit char --split ${SPLIT} \
     --w2v-ckpt ${WAV2VEC2_CHECKPOINT_PATH} --w2v-dict-dir ${WAV2VEC2_DICT_DIR} \
     --raw-manifest ${EVAL_OUTPUT_ROOT}/eval_16khz.tsv --asr-dir ${EVAL_OUTPUT_ROOT}/asr
   ```

   - MCD/MSD:
   ```bash
   python -m examples.speech_synthesis.evaluation.eval_sp \
     ${EVAL_OUTPUT_ROOT}/eval.tsv --mcd --msd
   ```

   - F0 metrics:
   ```bash
   python -m examples.speech_synthesis.evaluation.eval_f0 \
     ${EVAL_OUTPUT_ROOT}/eval.tsv --gpe --vde --ffe
   ```

## Results

| Architecture | Parameters | Test MCD | Model |
|---|---|---|---|
| tts_transformer | 54M | 3.8 | [Download](https://dl.fbaipublicfiles.com/fairseq/s2/ljspeech_transformer_phn.tar) |
| fastspeech2 | 41M | 3.8 | [Download](https://dl.fbaipublicfiles.com/fairseq/s2/ljspeech_fastspeech2_phn.tar) |