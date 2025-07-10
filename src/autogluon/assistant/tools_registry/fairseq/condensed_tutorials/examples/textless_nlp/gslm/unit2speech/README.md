# Condensed: Unit to Speech Model (unit2speech)

Summary: This tutorial introduces the unit2speech model, a modified Tacotron2 implementation that synthesizes speech from discrete speech units. It covers pre-trained models using various upstream units (Log Mel Filterbank, Modified CPC, HuBERT Base, wav2vec 2.0) with different clustering configurations. The implementation requires specific Python packages, model checkpoints, code dictionaries, and a Waveglow vocoder. The tutorial provides complete inference code with key parameters for speech synthesis, emphasizing that quantized audio must match the training units. This resource helps with implementing text-to-speech systems using discrete speech units and the Tacotron2 architecture.

*This is a condensed version that preserves essential implementation details and context.*

# Unit to Speech Model (unit2speech)

## Overview
The unit2speech model is a modified Tacotron2 model that synthesizes speech from discrete speech units. All models are trained on quantized LJSpeech dataset.

## Pre-trained Models

Various models are available with different upstream units and clustering configurations (KM50, KM100, KM200):
- Log Mel Filterbank
- Modified CPC
- HuBERT Base
- wav2vec 2.0 Large

## Implementation Details

### Requirements
```
pip install librosa unidecode inflect
```

### Required Files
- Unit2speech model checkpoint
- Code dictionary file
- [Waveglow vocoder checkpoint](https://dl.fbaipublicfiles.com/textless_nlp/gslm/waveglow_256channels_new.pt)

### Inference Code
```python
PYTHONPATH=${FAIRSEQ_ROOT}:${FAIRSEQ_ROOT}/examples/textless_nlp/gslm/unit2speech python ${FAIRSEQ_ROOT}/examples/textless_nlp/gslm/unit2speech/synthesize_audio_from_units.py \
    --tts_model_path $TTS_MODEL_PATH \
    --quantized_unit_path $QUANTIZED_UNIT_PATH \
    --out_audio_dir $OUT_DIR \
    --waveglow_path  $WAVEGLOW_PATH \
    --code_dict_path $CODE_DICT_PATH \
    --max_decoder_steps 2000
```

### Key Parameters
- `tts_model_path`: Path to the unit2speech model
- `quantized_unit_path`: Path to the quantized audio file
- `out_audio_dir`: Directory to save synthesized audio
- `waveglow_path`: Path to the Waveglow checkpoint
- `code_dict_path`: Path to the code dictionary
- `max_decoder_steps`: Maximum number of decoder steps (default: 2000)

> **Important**: The quantized audio must use the same units as the unit2speech model was trained with.