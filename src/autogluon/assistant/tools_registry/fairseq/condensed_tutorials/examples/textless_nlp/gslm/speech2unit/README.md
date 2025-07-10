# Condensed: Speech to Unit Model (speech2unit)

Summary: This tutorial explains the speech2unit model implementation, which quantizes speech using K-means clustering over acoustic representations. It helps with converting speech to discrete units by providing code for training K-means models and quantizing audio with those models. Key features include support for multiple acoustic representations (Log-Mel Filterbank, CPC, HuBERT, Wav2Vec 2.0), various cluster sizes (50/100/200), and a complete pipeline with pretrained models. The tutorial provides Python code examples for both training K-means models and applying them to quantize speech, along with parameter explanations and required file formats.

*This is a condensed version that preserves essential implementation details and context.*

# Speech to Unit Model (speech2unit)

## Overview
The speech2unit model quantizes speech by learning K-means clustering over acoustic representations, using either Log-Mel Filterbank or pretrained acoustic models.

## Pretrained Models

### Acoustic Models
- [Modified CPC](https://dl.fbaipublicfiles.com/textless_nlp/gslm/cpc/cpc_big_ll6kh_top_ctc.pt)
- [HuBERT-Base](https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt)
- [Wav2Vec 2.0-Base](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_vox_new.pt)

### Quantization Models
Various pretrained K-means models are available with different configurations:
- Log Mel Filterbank + KM50/100/200
- Modified CPC + KM50/100/200
- HuBERT Base + KM50/100/200
- wav2vec 2.0 Large + KM50/100/200

## Implementation

### 1. Learn K-means clustering model
```python
PYTHONPATH=. python examples/textless_nlp/gslm/speech2unit/clustering/cluster_kmeans.py \
    --num_clusters $N_CLUSTERS \
    --feature_type $TYPE \
    --checkpoint_path $CKPT_PATH \
    --layer $LAYER \
    --manifest_path $MANIFEST \
    --out_kmeans_model_path $KM_MODEL_PATH
```

### 2. Quantize using the learned clusters
```python
python examples/textless_nlp/gslm/speech2unit/clustering/quantize_with_kmeans.py \
    --feature_type $TYPE \
    --kmeans_model_path $KM_MODEL_PATH \
    --acoustic_model_path $CKPT_PATH \
    --layer $LAYER \
    --manifest_path $MANIFEST \
    --out_quantized_file_path $OUT_QUANTIZED_FILE \
    --extension ".flac"
```

### Important Parameters
- `N_CLUSTERS`: Number of clusters for K-means
- `TYPE`: Feature type (logmel/cpc/hubert/w2v2)
- `LAYER`: Layer of acoustic model to extract features from
- `CKPT_PATH`: Path to pretrained acoustic model
- `KM_MODEL_PATH`: Output path for K-means model

### Manifest File Format
```
<path_of_root_directory_containing_audio_files>
<relative_path_of_audio_file_1>\t<number_of_frames_1>
<relative_path_of_audio_file_2>\t<number_of_frames_1>
...
```