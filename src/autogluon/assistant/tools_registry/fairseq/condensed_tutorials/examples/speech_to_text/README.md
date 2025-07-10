# Condensed: Speech-to-Text (S2T) Modeling

Summary: This tutorial covers Fairseq's speech-to-text (S2T) modeling implementation, teaching how to prepare audio data using TSV manifests with pre-computed features or raw audio files, configure YAML files for tokenization and transforms, and train models using the speech_to_text task. It demonstrates batch and interactive inference workflows for ASR and speech translation tasks. Key functionalities include on-the-fly CMVN application, SpecAugment, temperature-based resampling, and support for multiple S2T tasks (ASR, ST, SimulST) with examples for LibriSpeech, MuST-C, CoVoST 2, and Multilingual TEDx datasets.

*This is a condensed version that preserves essential implementation details and context.*

# Speech-to-Text (S2T) Modeling in Fairseq

## Data Preparation

S2T modeling requires:
- TSV manifest files (one per dataset split) containing:
  - Source speech features/audio paths
  - Target text
  - Optional information (source text, speaker id, etc.)
- Speech features can be:
  - Pre-computed NumPy files
  - WAV/FLAC audio files (features extracted on-the-fly)
  - Packed in uncompressed ZIP files for I/O performance
- YAML configuration file for:
  - Target text tokenizer and dictionary path
  - Feature transforms (CMVN, SpecAugment)
  - Temperature-based resampling

## Model Training

```bash
fairseq-train \
  --task speech_to_text \
  --arch <model_architecture> \
  --config-yaml <config_yaml_filename> \
  [other training parameters]
```

## Inference & Evaluation

```bash
# For batch inference/evaluation
fairseq-generate \
  --task speech_to_text \
  --config-yaml <config_yaml_filename> \
  [other parameters]

# For interactive inference
fairseq-interactive \
  --task speech_to_text \
  --config-yaml <config_yaml_filename> \
  [other parameters]
```

The interactive console accepts audio paths (one per line) as input.

## Examples

- ASR on LibriSpeech
- ST on MuST-C
- ST on CoVoST 2
- ST on Multilingual TEDx
- Simultaneous ST on MuST-C

## Implementation Notes

- CMVN is applied on-the-fly (defined in config YAML) rather than during data preparation
- Interactive decoding is supported for both ASR and ST tasks
- The framework supports various speech-to-text tasks including ASR, ST, and SimulST