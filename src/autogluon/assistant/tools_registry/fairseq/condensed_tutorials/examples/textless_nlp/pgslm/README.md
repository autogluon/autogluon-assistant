# Condensed: Text-Free Prosody-Aware Generative Spoken Language Modeling

Summary: This tutorial covers implementing "Text-Free Prosody-Aware Generative Spoken Language Modeling" (pGSLM), a system that models speech directly from discrete units without text. It details: (1) data preprocessing techniques including HuBERT unit extraction and F0 quantization, (2) training multi-stream transformer language models with separate streams for units, duration, and F0, (3) evaluation methods measuring prosody correlation and continuation quality, and (4) sampling procedures with temperature controls for each stream. The code supports both continuous and quantized prosody representations, with pre-trained models and vocoders available for speech generation. Key functionalities include prosody-aware speech modeling, F0 processing, and configurable sampling parameters.

*This is a condensed version that preserves essential implementation details and context.*

# Text-Free Prosody-Aware Generative Spoken Language Modeling

This guide covers implementation details for reproducing the paper "Text-Free Prosody-Aware Generative Spoken Language Modeling" ([arxiv](https://arxiv.org/abs/2109.03264)).

## Additional Requirements

```bash
pip install AMFM-decompy SoundFile scipy sklearn torchaudio npy-append-array
```

## Data Preprocessing

### 1. Prepare Unit Pseudo-Text Transcriptions

First, create manifest files:
```bash
mkdir manifests/
python examples/wav2vec/wav2vec_manifest.py --valid-percent=0.0 $DATA_PATH --dest=manifests/train/
```

Next, quantize the dataset using HuBERT-base-ls960 model and kmeans-100 quantizer:
```bash
python examples/textless_nlp/gslm/speech2unit/clustering/quantize_with_kmeans.py \
    --feature_type hubert \
    --kmeans_model_path km.bin \
    --acoustic_model_path hubert_base_ls960.pt \
    --layer 6 \
    --manifest_path manifests/train/train.tsv \
    --out_quantized_file_path manifests/train/units
```

Join units with manifest:
```bash
python examples/textless_nlp/pgslm/scripts/join_units_manifest.py --manifest=manifests/train/train.tsv --units=manifests/train/units --output=train.txt
```

### 2. Preprocess Data for pGSLM

First, obtain F0 quantization bins:
```bash
bash examples/textless_nlp/pgslm/scripts/prepare_f0_quantization.sh \
  vocoder_train.txt <sample_rate> 32 <preprocessed_dir> <output_prefix>
```

This generates:
- `<output_prefix>.f0_stat.pt`: speaker-level F0 statistics
- `<output_prefix>_mean_norm_log_f0_bin.th`: quantized F0 for data preparation

Next, prepare the pGSLM data:
```bash
bash examples/textless_nlp/pgslm/scripts/prepare_data.sh \
  train.txt valid.txt test.txt <n_unit> <hop_size> <sample_rate> \
  <preprocessed_dir>/<output_prefix>_mean_norm_log_f0_bin.th <preprocessed_dir>
```

**Important note:** For large datasets, distribute F0 computation using `--nshards=x` and `--rank=z` in `preprocess_f0.py`, and set `--nshards_list=x` in `prepare_data.py`.

## Training Multi-Stream Transformer Unit Language Model (MS-TLM)

Example training command:
```bash
fairseq-train data_config.json \
  --task=speech_unit_modeling \
  --arch="transformer_ulm_tiny" \
  --criterion=speech_unit_lm_criterion \
  --share-decoder-input-output-embed \
  --dropout=0.1 \
  --attention-dropout=0.1 \
  --optimizer="adam" \
  --adam-betas="(0.9, 0.98)" \
  --clip-norm=1.0 \
  --lr=0.0005 \
  --lr-scheduler="inverse_sqrt" \
  --warmup-updates=4000 \
  --warmup-init-lr=1e-07 \
  --tokens-per-sample=3072 \
  --max-tokens=3072 \
  --update-freq=4 \
  --max-epoch=70 \
  --num-workers=0 \
  --skip-invalid-size-inputs-valid-test \
  --loss-weights="1.0;0.5;0.0" \
  --ignore-f0-input \
  --checkpoint-activations \
  --fp16 \
  --max-target-positions=4096 \
  --stream-shifts="1,1" \
  --log-f0 --normalize-f0-mean --interpolate-f0 \
  --ignore-unused-valid-subsets \
  --discrete-duration --discrete-f0
```

### Key Configuration Parameters

- **Architecture options**:
  - `transformer_ulm_tiny`: 2 layers, 1 head, 64 dim (debugging)
  - `transformer_ulm`: 6 layers, 8 heads, 512/2048 dim (base model)
  - `transformer_ulm_big`: 12 layers, 16 heads, 1024/4096 dim (largest model)

- **Stream configuration**:
  - `loss-weights`: Weights for unit, duration, and F0 streams (e.g., "1.0;0.5;0.0")
  - `stream-shifts`: Relative shifts of prosodic streams (duration,F0) to unit stream
  - `ignore-duration-input`/`ignore-f0-input`: Zero out corresponding input streams
  - `discrete-duration`/`discrete-f0`: Enable quantization of duration and F0

- **F0 processing**:
  - `log_f0`: Model F0 in log-space
  - `normalize-f0-mean`/`normalize-f0-std`: Per-speaker normalization
  - `interpolate-f0`: Interpolate F0 in unvoiced regions

- **Masking parameters**:
  - `mask-dur-prob`, `mask-f0-prob`: Probability of masking individual steps
  - `mask-dur-seg-prob`, `mask-f0-seg-prob`, `mask-unit-seg-prob`: Probability of masking spans
  - `mask-unit-seg-leng`: Length of masked spans

# Pre-trained Models and Evaluation

## MS-TLM Pre-trained Models

Four best-performing models from the paper are available, trained on Hubert-100 transcripts of LibriLight-6K dataset:

| | Continuous prosody | Quantized prosody |
|-------------------|--------------------|-------------------|
| No prosody input | [download](https://dl.fbaipublicfiles.com/textless_nlp/pgslm/ulm_checkpoints/continuous_no_prosody_shift_1_1.pt) | [download](https://dl.fbaipublicfiles.com/textless_nlp/pgslm/ulm_checkpoints/discrete_no_prosody_shift_1_1.pt) |
| Has prosody input | [download](https://dl.fbaipublicfiles.com/textless_nlp/pgslm/ulm_checkpoints/continuous_prosody_shift_1_1.pt) | [download](https://dl.fbaipublicfiles.com/textless_nlp/pgslm/ulm_checkpoints/discrete_prosody_shift_1_1.pt) |

**Optimal sampling parameters** (T-token, T-duration, T-f0):

| | Continuous prosody | Quantized prosody |
|-------------------|--------------------|-------------------|
| No prosody input | 0.7, 0.125, 0.0003125 | 0.7, 0.25, 0.5 |
| Has prosody input | 0.7, 0.125, 0.00125 | 0.7, 0.25, 0.7 |

## Vocoder

| Units | Prosody | F0 stats | Checkpoint | Config |
|-------------------|---------|--------------|------------|--------|
| HuBERT-base-ls960, kmeans-100 | [Quantized 32 bins](https://dl.fbaipublicfiles.com/textless_nlp/pgslm/vocoder/blizzard2013/mean_norm_log_f0_seg_bin.th) | [download](https://dl.fbaipublicfiles.com/textless_nlp/pgslm/vocoder/blizzard2013/f0_stats.pt) | [download](https://dl.fbaipublicfiles.com/textless_nlp/pgslm/vocoder/blizzard2013/naive_quant_32_norm_log_seg_hubert/checkpoint.pt) | [download](https://dl.fbaipublicfiles.com/textless_nlp/pgslm/vocoder/blizzard2013/naive_quant_32_norm_log_seg_hubert/config.json) |
| HuBERT-base-ls960, kmeans-100 | Continuous | [download](https://dl.fbaipublicfiles.com/textless_nlp/pgslm/vocoder/blizzard2013/f0_stats.pt) | [download](https://dl.fbaipublicfiles.com/textless_nlp/pgslm/vocoder/blizzard2013/mean_norm_log_f0_hubert/checkpoint.pt) | [download](https://dl.fbaipublicfiles.com/textless_nlp/pgslm/vocoder/blizzard2013/mean_norm_log_f0_hubert/config.json) |

## Evaluating a Trained Model

Evaluation uses the `eval/cont_metrics.py` script with several metrics:

### Teacher-forced Metrics

```bash
SET=valid
CHECKPOINT_PATH=discrete_prosody_shift_1_1.pt
DATA=data_config.json

python examples/textless_nlp/pgslm/eval/cont_metrics.py $DATA \
  --metric=teacher_force_everything \
  --path=$CHECKPOINT_PATH \
  --batch-size=16 \
  --fp16 \
  --seed=111 \
  --eval-subset=$SET \
  --f0-discretization-bounds=mean_norm_log_f0_seg_bin.th --dequantize-prosody 
```

The `--f0-discretization-bounds` and `--dequantize-prosody` parameters are specific for quantized-prosody models, signaling that prosody streams must be decoded into the continuous domain before calculating correlation.

### Consistency (Correlation) Metrics

Estimates correlation between mean F0 values in the prompt and generated continuation:

```bash
T_F0=0.7
EXPLOSION=20
SET=test
CHECKPOINT_PATH=discrete_prosody_shift_1_1.pt
DATA=data_config.json

python examples/textless_nlp/pgslm/eval/cont_metrics.py $DATA \
    --prefix-length=150 \
    --metric=correlation \
    --path=$CHECKPOINT_PATH \
    --batch-size=16 \
    --fp16 \
    --seed=111 \
    --teacher-force-tokens \
    --teacher-force-duration  \
    --min-length=300  \
    --batch-explosion-rate=$EXPLOSION \
    --T-f0=$T_F0 \
    --eval-subset=$SET \
    --f0-discretization-bounds=mean_norm_log_f0_seg_bin.th \
    --dequantize-prosody --n-workers=8
```

**Key parameters:**
- `--teacher-force-tokens`, `--teacher-force-duration`, `--teacher-force-f0`: Calculate correlations along each stream while fixing others to ground truth
- `T-f0`, `T-duration`, `T-token`: Per-stream temperatures/scaling parameters
- `min-length`: Filters sequences shorter than specified duration units
- `prefix-length`: Specifies duration units to use as prompt
- `n-workers`: Speeds up computation by distributing across GPUs

### Correctness (Continuation) and Expressiveness (Std) Metrics

Calculate minMAE and Std for the log-F0 stream:

```bash
DATA=data_config.json
EXPLOSION=20
SET=test
CHECKPOINT_PATH=discrete_prosody_shift_1_1.pt
T_F0=0.7

python examples/textless_nlp/pgslm/eval/cont_metrics.py $DATA \
  --prefix-length=150 \
  --metric=continuation \
  --path=$CHECKPOINT_PATH \
  --batch-size=16 \
  --fp16 \
  --seed=111 \
  --batch-explosion-rate=$EXPLOSION \
  --teacher-force-tokens \
  --teacher-force-duration \
  --T-f0=$T_F0 \
  --eval-subset=$SET \
  --f0-discretization-bounds=mean_norm_log_f0_seg_bin.th --dequantize-prosody
```

**Cont Word BLEU** evaluation follows the protocol from Lakhotia et al. (2021).

# Sampling from a Trained Model

## Basic Sampling Command

```bash
CHECKPOINT_PATH=checkpoints/checkpoint_best.pt
DATASET=examples/textless_nlp/pgslm/repro/dataset/data_config.json 
python examples/textless_nlp/pgslm/sample/sample.py $DATASET \
  --output=$SAMPLES \
  --path=$CHECKPOINT_PATH \
  --sampling \
  --T-token=0.7 \
  --T-duration=0.25 \
  --T-f0=0.7 \
  --max-length=500 \
  --prefix-length=150 \
  --subset=valid \
  --seed=1 \
  --match-duration \
  --code-type=hubert \
  --batch-explosion-rate=2
```


...(truncated)