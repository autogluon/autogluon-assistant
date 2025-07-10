# Condensed: Dialogue Unit-to-Speech Decoder for dGSLM

Summary: This tutorial demonstrates how to use the unit2speech decoder for dGSLM, which converts discrete speech units to audio using a HiFi-GAN vocoder trained on the Fisher dataset. It covers both command-line and interactive Python implementations for generating stereo waveforms from discrete unit sequences. The tutorial provides specific checkpoint URLs, explains input format requirements, and shows how to load the vocoder, decode unit sequences to audio waveforms, and play the resulting audio. This knowledge is valuable for implementing speech synthesis from discrete units in dialogue systems or speech processing applications.

*This is a condensed version that preserves essential implementation details and context.*

# Dialogue Unit-to-Speech Decoder for dGSLM

## Overview
The unit2speech decoder uses a discrete unit-based HiFi-GAN vocoder trained on the Fisher dataset.

## Model Checkpoint
```
HiFi-GAN vocoder: https://dl.fbaipublicfiles.com/textless_nlp/dgslm/checkpoints/hifigan/hifigan_vocoder
Config: https://dl.fbaipublicfiles.com/textless_nlp/dgslm/checkpoints/hifigan/config.json
```

## Usage

### Command Line Decoding
```bash
python examples/textless_nlp/dgslm/vocoder_hifigan/generate_stereo_waveform.py \
    --in-file $INPUT_CODE_FILE \
    --vocoder $VOCODER_PATH \
    --vocoder-cfg $VOCODER_CONFIG \
    --results-path $OUTPUT_DIR
```

**Input format requirements:**
```
{'audio': 'file_1', 'unitA': '8 8 ... 352 352', 'unitB': '217 8 ... 8 8'}
{'audio': 'file_2', 'unitA': '5 5 ... 65 65', 'unitB': '6 35 ... 8 9'}
...
```

### Interactive Decoding
```python
# Load the Hifigan vocoder
from examples.textless_nlp.dgslm.dgslm_utils import HifiganVocoder
decoder = HifiganVocoder(
    vocoder_path = "/path/to/hifigan_vocoder",
    vocoder_cfg_path = "/path/to/config.json",
)

# Decode the units to waveform
codes = [
    '7 376 376 133 178 486 486 486 486 486 486 486 486 2 486',
    '7 499 415 177 7 7 7 7 7 7 136 136 289 289 408',
]
wav = decoder.codes2wav(codes)
# > array of shape (2, 4800)

# Play the waveform
import IPython.display as ipd
ipd.Audio(wav, rate=16_000)
```