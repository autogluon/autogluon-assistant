# Condensed: [[Back]](..)

Summary: This tutorial demonstrates implementing Transformer models for speech synthesis using the VCTK English corpus. It covers essential techniques for TTS systems including: data preparation with denoising and silence trimming, feature extraction with phoneme inputs (IPA vocabulary with G2P conversion), quality filtering using SNR and CER thresholds, and model training. The guide helps with preprocessing speech data, creating feature manifests, and implementing transformer-based TTS models. Key functionalities include audio denoising, voice activity detection, phoneme conversion, and quality-based sample filtering, culminating in a 54M parameter transformer model achieving 3.4 MCD on test data.

*This is a condensed version that preserves essential implementation details and context.*

# VCTK Speech Synthesis

This guide covers implementing Transformer models for speech synthesis using the VCTK English speech corpus.

## Data Preparation

1. Download and prepare manifests:
```bash
python -m examples.speech_synthesis.preprocessing.get_vctk_audio_manifest \
  --output-data-root ${AUDIO_DATA_ROOT} \
  --output-manifest-root ${AUDIO_MANIFEST_ROOT}
```

2. Denoise audio and trim silence:
```bash
for SPLIT in dev test train; do
    python -m examples.speech_synthesis.preprocessing.denoise_and_vad_audio \
      --audio-manifest ${AUDIO_MANIFEST_ROOT}/${SPLIT}.audio.tsv \
      --output-dir ${PROCESSED_DATA_ROOT} \
      --denoise --vad --vad-agg-level 3
done
```

3. Filter by CER (Character Error Rate):
   - Run ASR model with `--eval-target` and `--err-unit char` flags
   - CER results will be saved to `${EVAL_OUTPUT_ROOT}/uer_cer.${SPLIT}.tsv`

4. Extract features and create configuration:
```bash
python -m examples.speech_synthesis.preprocessing.get_feature_manifest \
  --audio-manifest-root ${PROCESSED_DATA_ROOT} \
  --output-root ${FEATURE_MANIFEST_ROOT} \
  --ipa-vocab --use-g2p \
  --snr-threshold 15 \
  --cer-threshold 0.1 --cer-tsv-path ${EVAL_OUTPUT_ROOT}/uer_cer.${SPLIT}.tsv
```

## Key Implementation Details
- Uses phoneme inputs (`--ipa-vocab --use-g2p`)
- Filters samples with SNR threshold of 15
- Applies CER threshold of 10%
- For training, inference, and evaluation, refer to the LJSpeech example documentation

## Results

| Architecture | Parameters | Test MCD | Model |
|---|---|---|---|
| tts_transformer | 54M | 3.4 | [Download](https://dl.fbaipublicfiles.com/fairseq/s2/vctk_transformer_phn.tar) |