# Condensed: [[Back]](..)

Summary: This tutorial demonstrates how to prepare and process the Common Voice speech corpus for speech synthesis tasks. It covers implementation techniques for: downloading and converting audio data, denoising and silence trimming with VAD, filtering samples by Character Error Rate (CER) and Signal-to-Noise Ratio (SNR), and extracting features with phoneme inputs. The tutorial helps with preprocessing speech data, creating audio manifests, and preparing datasets for training TTS models. Key functionalities include audio conversion, noise reduction, voice activity detection, quality filtering, and feature extraction, with a specific example of a trained English TTS transformer model achieving 3.8 MCD on test data.

*This is a condensed version that preserves essential implementation details and context.*

# Common Voice

[Common Voice](https://commonvoice.mozilla.org/en/datasets) is a public domain speech corpus with 11.2K hours of read speech in 76 languages (v7.0).

## Data Preparation

1. **Download and prepare manifests**:
   ```bash
   python -m examples.speech_synthesis.preprocessing.get_common_voice_audio_manifest \
     --data-root ${DATA_ROOT} \
     --lang ${LANG_ID} \
     --output-manifest-root ${AUDIO_MANIFEST_ROOT} --convert-to-wav
   ```

2. **Denoise audio and trim silence**:
   ```bash
   for SPLIT in dev test train; do
       python -m examples.speech_synthesis.preprocessing.denoise_and_vad_audio \
         --audio-manifest ${AUDIO_MANIFEST_ROOT}/${SPLIT}.audio.tsv \
         --output-dir ${PROCESSED_DATA_ROOT} \
         --denoise --vad --vad-agg-level 2
   done
   ```
   This generates a new audio TSV manifest with updated paths and SNR values.

3. **Filter by CER**: Run ASR model evaluation with `--eval-target` and `--err-unit char` flags to compute CER on reference audio.

4. **Extract features and create configuration**:
   ```bash
   python -m examples.speech_synthesis.preprocessing.get_feature_manifest \
     --audio-manifest-root ${AUDIO_MANIFEST_ROOT} \
     --output-root ${FEATURE_MANIFEST_ROOT} \
     --ipa-vocab --lang ${LANG_ID} \
     --snr-threshold 15 \
     --cer-threshold 0.1 --cer-tsv-path ${EVAL_OUTPUT_ROOT}/uer_cer.${SPLIT}.tsv
   ```
   - Uses phoneme inputs (`--ipa-vocab`)
   - Filters samples with SNR threshold of 15
   - Filters samples with CER threshold of 10%

## Results

| Language | Speakers | --arch | Params | Test MCD | Model |
|---|---|---|---|---|---|
| English | 200 | tts_transformer | 54M | 3.8 | [Download](https://dl.fbaipublicfiles.com/fairseq/s2/cv4_en200_transformer_phn.tar) |

For training, inference, and evaluation procedures, refer to the [LJSpeech example](../docs/ljspeech_example.md).