# Condensed: wav2vec 2.0

Summary: This tutorial covers Wav2Vec 2.0 implementation for self-supervised speech representation learning. It details how to work with pre-trained models (Base, Large, XLSR-53, multilingual variants), train new models, and fine-tune them for speech recognition using CTC. Key functionalities include: preparing audio data manifests, training models with different architectures (Transformer or Conformer backbones), fine-tuning with CTC loss, evaluating models with various decoders, extracting embeddings, and using vector quantization (VQ-Wav2Vec). The tutorial also demonstrates integration with Hugging Face Transformers and provides complete command-line examples for training on GPUs or TPUs.

*This is a condensed version that preserves essential implementation details and context.*

# Pre-trained Wav2Vec 2.0 Models

## Available Pre-trained Models

### Base and Large Models on LibriSpeech
- **Wav2Vec 2.0 Base**: Available with no finetuning, 10min, 100h, and 960h finetuning splits
- **Wav2Vec 2.0 Large**: Available with no finetuning, 10min, 100h, and 960h finetuning splits

### Large Models on Libri-Light (LV-60)
- **Standard Large**: No finetuning, 10min, 100h, and 960h versions
- **Large Conformer variants**: 
  - rel_pos: No finetuning, 100h, 960h versions
  - rope: No finetuning, 100h, 960h versions
- **Large + Self Training**: 10min, 100h, and 960h versions

### Multi-dataset Large Model
- **LV-60 + CV + SWBD + FSH**: No finetuning, 960h Librispeech, and 300h Switchboard versions

## Multilingual Models

### XLSR-53
- **Architecture**: Large
- **Training data**: 56k hours across 53 languages
- **Datasets**: MLS, CommonVoice, BABEL

### Finetuned Multilingual Models
- **LV-60 finetuned on CommonVoice**: 26 languages with Espeak phonemizer
- **XLSR-53 finetuned on CommonVoice**: 26 languages with Espeak phonemizer

## Implementation Note
All models are downloadable via the links in the original tables. Updated models are marked with asterisks (* updated Oct. 24, 2020, ** updated Nov. 13, 2021).

# XLSR-53 Pre-trained Models

## Available Models

| Base Model | Training Data | Languages | Phonemizer | Model | Dictionary |
|------------|---------------|-----------|------------|-------|------------|
| XLSR-53 | CommonVoice | 21 | Phonetisaurus | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/zero_shot/phonetisaurus_21lang_m10.pt) | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/zero_shot/phonetisaurus_dict.txt) |
| XLSR-53 | CommonVoice, BABEL | 21, 19 | Phonetisaurus | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/zero_shot/phonetisaurus_40lang_m10.pt) | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/zero_shot/phonetisaurus_40lang.dict.txt) |

## Important Note
Both models use IPA symbols, but there are subtle differences between the phonemized transcriptions from the two phonemizers. For best results, use the corresponding model that matches your phonemization method.

# wav2vec 2.0 Implementation Guide

## Overview
wav2vec 2.0 is a framework for self-supervised learning of speech representations, described in [Baevski et al., 2020](https://arxiv.org/abs/2006.11477) and extended in subsequent research.

## Training a New Model

### Prepare Training Data Manifest

1. Install required library:
```shell
pip install soundfile
```

2. Generate manifest files:
```shell
python examples/wav2vec/wav2vec_manifest.py /path/to/waves --dest /manifest/path --ext $ext --valid-percent $valid
```
- `$ext`: audio format (flac, wav, etc.)
- `$valid`: percentage of data for validation (e.g., 0.01)

### Training Models

#### Base Model (Librispeech)
```shell
fairseq-hydra-train \
    task.data=/path/to/data \
    --config-dir /path/to/fairseq-py/examples/wav2vec/config/pretraining \
    --config-name wav2vec2_base_librispeech
```
*Note*: Simulate 64 GPUs with `distributed_training.distributed_world_size=k` `+optimization.update_freq='[x]'` where x = 64/k

#### Large Model (Libri-light)
```shell
fairseq-hydra-train \
    task.data=/path/to/data \
    --config-dir /path/to/fairseq-py/examples/wav2vec/config/pretraining \
    --config-name wav2vec2_large_librivox
```
*Note*: Simulate 128 GPUs with `distributed_training.distributed_world_size=k` `+optimization.update_freq='[x]'` where x = 128/k

#### Conformer Backbone Models
To use conformer layers instead of transformer layers:

```shell
fairseq-hydra-train \
    task.data=/path/to/data \
    --config-dir /path/to/fairseq-py/examples/wav2vec/config/pretraining \
    --config-name wav2vec2_conformer_base_librispeech \
    --attn-type espnet --pos-enc-type ${POS_ENC_TYPE}
```

Where `${POS_ENC_TYPE}` can be:
- `abs`: absolute positional encoding
- `rope`: rotary positional encoding
- `rel_pos`: relative positional encoding

### Fine-tuning with CTC

1. Prepare labels (example for Librispeech):
```shell
python libri_labels.py /path/to/tsv --output-dir /output/dir --output-name train
```

2. Fine-tune on 100h of Librispeech:
```shell
fairseq-hydra-train \
    distributed_training.distributed_port=$PORT \
    task.data=/path/to/data \
    model.w2v_path=/path/to/model.pt \
    --config-dir /path/to/fairseq-py/examples/wav2vec/config/finetuning \
    --config-name base_100h
```

*Note*: Simulate 24 GPUs with `distributed_training.distributed_world_size=k` `+optimization.update_freq='[x]'` where x = 24/k

To use a language model during training, add:
```
+criterion.wer_args='[/path/to/kenlm, /path/to/lexicon, 2, -1]'
```
(requires flashlight python bindings)

### Evaluating a CTC Model

```shell
python examples/speech_recognition/infer.py /path/to/data --task audio_finetuning \
--nbest 1 --path /path/to/model --gen-subset dev_other --results-path /path/to/results --w2l-decoder kenlm \
--lm-model /path/to/kenlm.bin --lm-weight 2 --word-score -1 --sil-weight 0 --criterion ctc --labels ltr --max-tokens 4000000 \
--post-process letter
```

- For raw numbers: use `--w2l-decoder viterbi` and omit the lexicon
- For transformer LM: use `--w2l-decoder fairseqlm`

## Using wav2vec 2.0 with 🤗 Transformers

Available in Transformers library (v4.4+) with models on the [hub](https://huggingface.co/models?filter=wav2vec2).

```python
import soundfile as sf
import torch
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Load pretrained model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Load audio
librispeech_samples_ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
audio_input, sample_rate = sf.read(librispeech_samples_ds[0]["file"])

# Process audio
input_values = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_values

# Inference
logits = model(input_values).logits
predicted_ids = torch.argmax(logits, dim=-1)
```

# Speech Recognition with Wav2Vec Models - Implementation Guide

## Transcription and Fine-tuning

```python
# Basic transcription
transcription = processor.decode(predicted_ids[0])

# Fine-tuning example
target_transcription = "A MAN SAID TO THE UNIVERSE I EXIST"

# Encode labels for training
with processor.as_target_processor():
  labels = processor(target_transcription, return_tensors="pt").input_ids

# Compute loss and backpropagate
loss = model(input_values, labels=labels).loss
loss.backward()
```

## Wav2Vec Models

### Pre-trained Models

| Description | Dataset | Model |
|-------------|---------|-------|
| Wav2Vec large | [Librispeech](http://www.openslr.org/12) | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_large.pt) |

### Usage Example

```python
import torch
import fairseq

cp_path = '/path/to/wav2vec.pt'
model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
model = model[0]
model.eval()

wav_input_16khz = torch.randn(1,10000)
z = model.feature_extractor(wav_input_16khz)
c = model.feature_aggregator(z)
```

### Training Pipeline

1. **Prepare data manifest**:
   ```bash
   python examples/wav2vec/wav2vec_manifest.py /path/to/waves --dest /manifest/path --ext wav
   ```

2. **Train a wav2vec model**:
   ```bash
   python train.py /manifest/path --save-dir /model/path --num-workers 6 --fp16 --max-update 400000 \
   --save-interval 1 --no-epoch-checkpoints --arch wav2vec --task audio_pretraining \
   --min-lr 1e-06 --stop-min-lr 1e-09 --optimizer adam --lr 0.005 --lr-scheduler cosine \
   --conv-feature-layers [(512, 10, 5), (512, 8, 4), (512, 4, 2), (512, 4, 2), (512, 4, 2), (512, 1, 1), (512, 1, 1)] \
   --conv-aggregator-layers [(512, 2, 1), (512, 3, 1), (512, 4, 1), (512, 5, 1), (512, 6, 1), (512, 7, 1), (512, 8, 1), (512, 9, 1), (512, 10, 1), (512, 11, 1), (512, 12, 1), (512, 13, 1)] \
   --skip-connections-agg --residual-scale 0.5 --log-compression --warmup-updates 500 \
   --warmup-init-lr 1e-07 --criterion wav2vec --num-negatives 10 \
   --max-sample-size 150000 --max-tokens 1500000 --skip-invalid-size-inputs-valid-test
   ```

### TPU Training

**Using hydra on a v3-8**:
```bash
OMP_NUM_THREADS=1 fairseq-hydra-train \
  task.data=/manifest/path \
  --config-dir /PATH/TO/FAIRSEQ/examples/wav2vec/config/pretraining \
  --config-name wav2vec2_large_librivox_tpu.yaml
```

**For pod slices (v3-N with N > 8)**:
```bash
OMP_NUM_THREADS=1 fairseq-hydra-train \
  task.data=/manifest/path \
  --config-dir /PATH/TO/FAIRSEQ/examples/wav2vec/config/pretraining \
  --config-name wav2vec2_large_librivox_tpu-pod.yaml  # edit distributed-world-size accordingly
```

### Extract Embeddings
```bash
PYTHONPATH=/path/to/fairseq python examples/wav2vec/wav2vec_featurize.py \
  --input /path/to/task/waves --output /path/to/output \
  --model /model/path/checkpoint_best.pt --split train valid test
```

## VQ-Wav2Vec

### Pre-trained Models

| Description | Dataset | Model |
|-------------|---------|-------|
| vq-wav2vec Gumbel | Librispeech | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/vq-wav2vec.pt) |
| vq-wav2vec K-means | Librispeech | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/vq-wav2vec_kmeans.pt) |
| Roberta on K-means codes | Librispeech | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/bert_kmeans.tar) |

### Usage Example

```python
import torch
import fairseq

cp = torch.load('/path/to/vq-wav2vec.pt')
model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp])
model = model[0]
model.eval()

wav_input_16khz = torch.randn(1,10000)
z = model.feature_extractor(wav_input_16khz)
_, idxs = model.vector_quantizer.forward_idx(z)
# Output shape: [1, 60, 2] - 60 timesteps with 2 indexes for 2 groups
```

# Training and Using VQ-Wav2Vec Models


...(truncated)