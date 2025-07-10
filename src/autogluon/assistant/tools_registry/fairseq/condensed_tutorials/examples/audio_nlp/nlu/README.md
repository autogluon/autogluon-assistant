# Condensed: End-to-end NLU

Summary: This tutorial demonstrates implementing end-to-end spoken language understanding (SLU) models that predict intent directly from audio, eliminating ASR-related cascading errors. It covers creating fairseq datasets from the STOP dataset, training models using speech pretraining (Wav2Vec/HuBERT), and leveraging ASR pretraining for improved accuracy. The implementation uses fairseq's hydra-based configuration system and includes code for generating manifests, creating dictionaries, and model training. Key functionalities include utilizing pretrained speech models as foundations, achieving significant accuracy improvements through speech pretraining (36.54% to 68%+), and further enhancing performance with in-domain ASR pretraining.

*This is a condensed version that preserves essential implementation details and context.*

# End-to-end NLU Implementation Guide

## Overview
End-to-end spoken language understanding (SLU) predicts intent directly from audio using a single model, avoiding cascading errors from ASR and improving efficiency for on-device deployment.

## Dataset
- Main dataset: [STOP dataset](https://dl.fbaipublicfiles.com/stop/stop.tar.gz)
- Low-resource splits: [download link](http://dl.fbaipublicfiles.com/stop/low_resource_splits.tar.gz)

## Pretrained Models

### End-to-end NLU Models
| Speech Pretraining | ASR Pretraining | Test EM Accuracy | Test EM-Tree Accuracy | Link |
|-------------------|-----------------|------------------|------------------------|------|
| None | None | 36.54 | 57.01 | [link](https://dl.fbaipublicfiles.com/stop/end-to-end-nlu-none-none.pt) |
| Wav2Vec | None | 68.05 | 82.53 | [link](https://dl.fbaipublicfiles.com/stop/end-to-end-nlu-wav2vec-none.pt) |
| HuBERT | None | 68.40 | 82.85 | [link](https://dl.fbaipublicfiles.com/stop/end-to-end-nlu-hubert-none.pt) |
| HuBERT | STOP | 69.23 | 82.87 | [link](https://dl.fbaipublicfiles.com/stop/end-to-end-nlu-hubert-stop.pt) |

### ASR Models
Various pretrained ASR models are available with different speech pretraining (HuBERT/Wav2Vec) and training datasets (Librispeech/STOP/combined), with WER metrics for different test sets.

## Implementation Steps

### 1. Create Fairseq Datasets from STOP

Generate audio file manifests and label files:
```bash
python examples/audio_nlp/nlu/generate_manifests.py \
    --stop_root $STOP_DOWNLOAD_DIR/stop \
    --output $FAIRSEQ_DATASET_OUTPUT/
```

Generate fairseq dictionaries:
```bash
./examples/audio_nlp/nlu/create_dict_stop.sh $FAIRSEQ_DATASET_OUTPUT
```

### 2. Training an End-to-end NLU Model

First, download a wav2vec or hubert pretrained model from the fairseq repository.

Then train the model:
```bash
python fairseq_cli/hydra-train \
    --config-dir examples/audio_nlp/nlu/configs/ \
    --config-name nlu_finetuning \
    task.data=$FAIRSEQ_DATA_OUTPUT \
    model.w2v_path=$PRETRAINED_MODEL_PATH
```

## Key Implementation Details
- The system uses pretrained speech models (Wav2Vec or HuBERT) as a foundation
- Performance significantly improves with speech pretraining (36.54% → 68%+ accuracy)
- ASR pretraining on in-domain data (STOP) provides additional gains
- The implementation follows fairseq's hydra-based configuration system