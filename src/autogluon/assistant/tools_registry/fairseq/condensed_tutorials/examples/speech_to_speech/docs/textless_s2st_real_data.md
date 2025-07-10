# Condensed: Textless Speech-to-Speech Translation (S2ST) on Real Data

Summary: This tutorial explains how to implement Textless Speech-to-Speech Translation based on the Lee et al. 2021 paper. It provides code for a two-stage pipeline: speech normalization (converting speech to discrete units) and unit-to-waveform conversion using HiFi-GAN vocoders. The implementation leverages pre-trained models including mHuBERT Base (trained on multilingual data with L11 km1000 quantizer), language-specific vocoders, and speech normalizers with varying training data sizes. Key functionalities include audio preprocessing, discrete unit extraction, duration prediction, and waveform generation for English, Spanish, and French speech translation without using text.

*This is a condensed version that preserves essential implementation details and context.*

# Textless Speech-to-Speech Translation (S2ST) on Real Data

This guide covers implementation details for the paper "[Textless Speech-to-Speech Translation on Real Data (Lee et al. 2021)](https://arxiv.org/abs/2112.08352)".

## Pre-trained Models

### Key Components
- **mHuBERT Base**: Multilingual model trained on VoxPopuli (En, Es, Fr) with [L11 km1000 quantizer](https://dl.fbaipublicfiles.com/hubert/mhubert_base_vp_en_es_fr_it3_L11_km1000.bin)
- **Unit-based HiFi-GAN vocoders**: Available for English, Spanish, and French
- **Speech normalizers**: Available in 10min, 1hr, and 10hr training variants for each language

## Inference Pipeline

### 1. Speech Normalization

```bash
# Step 1: Format audio data
python examples/speech_to_speech/preprocessing/prep_sn_data.py \
  --audio-dir ${AUDIO_DIR} --ext ${AUDIO_EXT} \
  --data-name ${GEN_SUBSET} --output-dir ${DATA_DIR} \
  --for-inference

# Step 2: Run speech normalizer
python examples/speech_recognition/new/infer.py \
  --config-dir examples/hubert/config/decode/ \
  --config-name infer_viterbi \
  task.data=${DATA_DIR} \
  task.normalize=false \
  common_eval.results_path=${RESULTS_PATH}/log \
  common_eval.path=${DATA_DIR}/checkpoint_best.pt \
  dataset.gen_subset=${GEN_SUBSET} \
  '+task.labels=["unit"]' \
  +decoding.results_path=${RESULTS_PATH} \
  common_eval.post_process=none \
  +dataset.batch_size=1 \
  common_eval.quiet=True

# Step 3: Post-process output
python examples/speech_to_speech/preprocessing/prep_sn_output_data.py \
  --in-unit ${RESULTS_PATH}/hypo.units \
  --in-audio ${DATA_DIR}/${GEN_SUBSET}.tsv \
  --output-root ${RESULTS_PATH}
```

### 2. Unit-to-Waveform Conversion

```bash
python examples/speech_to_speech/generate_waveform_from_code.py \
  --in-code-file ${IN_CODE_FILE} \
  --vocoder ${VOCODER_CKPT} --vocoder-cfg ${VOCODER_CFG} \
  --results-path ${RESULTS_PATH} --dur-prediction
```

**Important Note**: Use `--dur-prediction` flag when generating audio from reduced unit sequences (with duplicate consecutive units removed).

## Technical Details

- The system uses mHuBERT layer 11 with 1000 units for discrete representation
- Speech normalizers are available in different training data sizes (10min, 1hr, 10hr)
- HiFi-GAN vocoders are language-specific and trained on LJSpeech (En) or CSS10 (Es, Fr)
- All models support multilingual processing (English, Spanish, French)