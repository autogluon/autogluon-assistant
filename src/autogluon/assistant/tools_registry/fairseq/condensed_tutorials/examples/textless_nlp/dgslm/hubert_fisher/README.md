# Condensed: Dialogue Speech-to-Unit Encoder for dGSLM: The Fisher HuBERT model

Summary: This tutorial provides implementation details for the Fisher HuBERT model, a dialogue speech-to-unit encoder for dGSLM. It covers how to encode audio into discrete units using a HuBERT model trained on the Fisher dataset with a 500-unit k-means quantizer. The tutorial offers both command-line and Python interfaces for processing stereo audio files, extracting features from layer 12 of the HuBERT model, and quantizing them into discrete units. Key functionalities include handling multi-channel audio, generating manifest files, and implementing the HubertTokenizer class for seamless audio-to-code conversion in dialogue processing tasks.

*This is a condensed version that preserves essential implementation details and context.*

# Dialogue Speech-to-Unit Encoder for dGSLM: Fisher HuBERT Model

## Model Checkpoints

| Component | Download Link |
|-----------|---------------|
| Fisher HuBERT model | [download](https://dl.fbaipublicfiles.com/textless_nlp/dgslm/checkpoints/hubert/hubert_fisher.pt) |
| k-means model | [download](https://dl.fbaipublicfiles.com/textless_nlp/dgslm/checkpoints/hubert/hubert_fisher_km_500.bin) |

## Implementation Details
- HuBERT model trained on Fisher dataset for 3 iterations
- k-means model with 500 units trained on layer 12 features of the HuBERT model

## Encoding Audio to Discrete Units

### Command-line Method
```bash
# First generate manifest file
python examples/wav2vec/wav2vec_manifest.py --valid-percent=0.0 $AUDIO_DIR --dest=$OUTPUT_DIR --ext=$EXTENSION

# Then encode each channel of stereo audio
for CHANNEL_ID in 1 2; do
    python examples/textless_nlp/gslm/speech2unit/clustering/quantize_with_kmeans.py \
        --feature_type hubert \
        --kmeans_model_path path/to/hubert_fisher_km_500.bin \
        --acoustic_model_path path/to/hubert_fisher.pt \
        --layer 12 \
        --manifest_path $MANIFEST_FILE \
        --out_quantized_file_path ${OUTPUT_FILE}-channel${CHANNEL_ID} \
        --extension $EXTENSION \
        --channel_id $CHANNEL_ID
done
```

### Python Interface
```python
from examples.textless_nlp.dgslm.dgslm_utils import HubertTokenizer

# Initialize tokenizer
encoder = HubertTokenizer(
    hubert_path = "/path/to/hubert_ckpt.pt",
    hubert_layer = 12,
    km_path = "path/to/km.bin"
)

# Encode audio file to units
path = "/path/to/stereo/audio.wav"
codes = encoder.wav2codes(path)
# Returns list of discrete unit sequences for each channel
```

## Key Parameters
- `hubert_layer = 12`: Extract features from layer 12 of HuBERT
- `km_path`: Path to k-means model with 500 units
- `channel_id`: For stereo audio processing (1 or 2)