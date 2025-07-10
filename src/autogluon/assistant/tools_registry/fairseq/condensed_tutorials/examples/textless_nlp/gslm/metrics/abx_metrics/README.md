# Condensed: ABX-based evaluation

Summary: This tutorial explains how to perform ABX-based evaluation for speech-to-unit models, focusing on computing ABX scores from quantized speech representations. It covers two key implementation steps: (1) extracting quantized features from audio test files using a pre-trained acoustic model and K-means clustering, and (2) computing ABX discrimination scores using libri-light's evaluation scripts. The tutorial provides complete code examples for feature extraction and ABX computation, with important configuration parameters for different model types (HuBERT, Wav2Vec2.0, CPC). This knowledge helps with implementing speech representation quality evaluation tasks, particularly for assessing phonetic discriminability in self-supervised speech models.

*This is a condensed version that preserves essential implementation details and context.*

# ABX-based Evaluation for Speech-to-Unit

This guide covers steps 3-4 of the ABX evaluation process, assuming you've already:
1. Trained an acoustic model
2. Learned K-means clustering for speech quantization

## Computing ABX

### Step 1: Dump Quantized Features

Extract quantized representations from test files:

```shell
TYPE="hubert"
LAYER=6
CKPT_PATH="<PATH_TO_HUBERT_MODEL_CHECKPOINT_FILE>"
KM_MODEL_PATH="<PATH_TO_PRETRAINED_KM_MODEL_FILE>"

SUBSET="dev-clean"
MANIFEST="<PATH_TO_MANIFEST_FOR_LS_DEV-CLEAN>"
DATA_DIR="<PATH_TO_DIR_TO_STORE_FEATURES>/$SUBSET"

PYTHONPATH=. python examples/textless_nlp/gslm/metrics/abx_metrics/dump_abx_feats.py \
    --feature_type $TYPE \
    --kmeans_model_path $KM_MODEL_PATH \
    --checkpoint_path $CKPT_PATH \
    --layer $LAYER \
    --manifest_path $MANIFEST \
    --out_dir_path $DATA_DIR \
    --extension ".flac"
```

### Step 2: Compute ABX Scores

Use libri-light's evaluation script:

```shell
LIBRILIGHT_ROOT="<PATH_TO_LIBRILIGHT>"

SUBSET="dev-clean"
DATA_DIR="<PATH_TO_DIR_TO_STORE_FEATURES>/$SUBSET"
ITEM_FILE_PATH="$LIBRILIGHT_ROOT/eval/ABX_data/$SUBSET.item"
OUT_DIR="<PATH_TO_DIR_TO_STORE_ABX_SCORES>/$SUBSET"

FILE_EXTENSION=".npy"
FEATURE_SIZE=0.02  # Model-dependent parameter

PYTHONPATH=$LIBRILIGHT_ROOT \
  python $LIBRILIGHT_ROOT/eval/eval_ABX.py \
    $DATA_DIR \
    $ITEM_FILE_PATH \
    --file_extension $FILE_EXTENSION \
    --feature_size $FEATURE_SIZE \
    --out $OUT_DIR \
    --mode "all"
```

## Important Configuration Notes

- **FEATURE_SIZE** depends on the model type:
  - HuBERT and Wav2Vec2.0: use `FEATURE_SIZE=0.02`
  - CPC and Log Mel: use `FEATURE_SIZE=0.01`
  
- Add `--cuda` flag for GPU acceleration

- Follow [libri-light's instructions](https://github.com/facebookresearch/libri-light) for installation and [ABX evaluation setup](https://github.com/facebookresearch/libri-light/tree/master/eval#abx)