# Condensed: Simultaneous Speech Translation (SimulST) on MuST-C

Summary: This tutorial demonstrates implementing simultaneous speech translation (SimulST) on the MuST-C English-German dataset using transformer wait-k models. It covers data preparation with feature extraction and vocabulary generation, ASR pretraining as initialization, and training simultaneous translation models with either wait-k or monotonic multihead attention approaches. The tutorial includes code for fixed pre-decision modules that make policy decisions at chunk boundaries, inference using SimulEval, and evaluation metrics for both translation quality (BLEU) and latency (AL, AP, DAL). Developers can use this to implement real-time speech translation systems with controllable quality-latency tradeoffs.

*This is a condensed version that preserves essential implementation details and context.*

# Simultaneous Speech Translation (SimulST) on MuST-C

This tutorial covers training and evaluating a transformer *wait-k* simultaneous model on MUST-C English-German Dataset.

## Data Preparation

```bash
# Install required packages
pip install pandas torchaudio sentencepiece

# Generate TSV manifests, features, vocabulary, and configurations
python examples/speech_to_text/prep_mustc_data.py \
  --data-root ${MUSTC_ROOT} --task asr \
  --vocab-type unigram --vocab-size 10000 \
  --cmvn-type global

python examples/speech_to_text/prep_mustc_data.py \
  --data-root ${MUSTC_ROOT} --task st \
  --vocab-type unigram --vocab-size 10000 \
  --cmvn-type global
```

## ASR Pretraining

```bash
fairseq-train ${MUSTC_ROOT}/en-de \
  --config-yaml config_asr.yaml --train-subset train_asr --valid-subset dev_asr \
  --save-dir ${ASR_SAVE_DIR} --num-workers 4 --max-tokens 40000 --max-update 100000 \
  --task speech_to_text --criterion label_smoothed_cross_entropy --report-accuracy \
  --arch convtransformer_espnet --optimizer adam --lr 0.0005 --lr-scheduler inverse_sqrt \
  --warmup-updates 10000 --clip-norm 10.0 --seed 1 --update-freq 8
```

A pretrained ASR checkpoint is available [here](https://dl.fbaipublicfiles.com/simultaneous_translation/must_c_v1_en_de_pretrained_asr).

## Simultaneous Speech Translation Training

### Wait-K with fixed pre-decision module
Fixed pre-decision makes simultaneous policy decisions at fixed chunk boundaries.

```bash
fairseq-train ${MUSTC_ROOT}/en-de \
  --config-yaml config_st.yaml --train-subset train_st --valid-subset dev_st \
  --save-dir ${ST_SAVE_DIR} --num-workers 8  \
  --optimizer adam --lr 0.0001 --lr-scheduler inverse_sqrt --clip-norm 10.0 \
  --criterion label_smoothed_cross_entropy \
  --warmup-updates 4000 --max-update 100000 --max-tokens 40000 --seed 2 \
  --load-pretrained-encoder-from ${ASR_SAVE_DIR}/checkpoint_best.pt \
  --task speech_to_text  \
  --arch convtransformer_simul_trans_espnet  \
  --simul-type waitk_fixed_pre_decision  \
  --waitk-lagging 3 \
  --fixed-pre-decision-ratio 7 \
  --update-freq 8
```

### Monotonic multihead attention with fixed pre-decision module

```bash
fairseq-train ${MUSTC_ROOT}/en-de \
  --config-yaml config_st.yaml --train-subset train_st --valid-subset dev_st \
  --save-dir ${ST_SAVE_DIR} --num-workers 8  \
  --optimizer adam --lr 0.0001 --lr-scheduler inverse_sqrt --clip-norm 10.0 \
  --warmup-updates 4000 --max-update 100000 --max-tokens 40000 --seed 2 \
  --load-pretrained-encoder-from ${ASR_SAVE_DIR}/${CHECKPOINT_FILENAME} \
  --task speech_to_text  \
  --criterion latency_augmented_label_smoothed_cross_entropy \
  --latency-weight-avg 0.1 \
  --arch convtransformer_simul_trans_espnet  \
  --simul-type infinite_lookback_fixed_pre_decision  \
  --fixed-pre-decision-ratio 7 \
  --update-freq 8
```

## Inference & Evaluation

[SimulEval](https://github.com/facebookresearch/SimulEval) is used for evaluation:

```bash
git clone https://github.com/facebookresearch/SimulEval.git
cd SimulEval
pip install -e .

simuleval \
    --agent ${FAIRSEQ}/examples/speech_to_text/simultaneous_translation/agents/fairseq_simul_st_agent.py
    --source ${SRC_LIST_OF_AUDIO}
    --target ${TGT_FILE}
    --data-bin ${MUSTC_ROOT}/en-de \
    --config config_st.yaml \
    --model-path ${ST_SAVE_DIR}/${CHECKPOINT_FILENAME} \
    --output ${OUTPUT} \
    --scores
```

### Preparing Evaluation Data

Generate wav list and text file for evaluation:

```bash
python ${FAIRSEQ}/examples/speech_to_text/seg_mustc_data.py \
  --data-root ${MUSTC_ROOT} --lang de \
  --split ${SPLIT} --task st \
  --output ${EVAL_DATA}
```

A prepared data directory is available [here](https://dl.fbaipublicfiles.com/simultaneous_translation/must_c_v1.0_en_de_databin.tgz).

### Configuration File Example

```yaml
bpe_tokenizer:
  bpe: sentencepiece
  sentencepiece_model: ABS_PATH_TO_SENTENCEPIECE_MODEL
global_cmvn:
  stats_npz_path: ABS_PATH_TO_GCMVN_FILE
input_channels: 1
input_feat_per_channel: 80
sampling_alpha: 1.0
specaugment:
  freq_mask_F: 27
  freq_mask_N: 1
  time_mask_N: 1
  time_mask_T: 100
  time_mask_p: 1.0
  time_wrap_W: 0
transforms:
  '*':
  - global_cmvn
  _train:
  - global_cmvn
  - specaugment
vocab_filename: spm_unigram10000_st.txt
```

### Pretrained Model

A pretrained wait-5 model with 280ms pre-decision is available [here](https://dl.fbaipublicfiles.com/simultaneous_translation/convtransformer_wait5_pre7).

Sample results on `tst-COMMON`:
```json
{
    "Quality": {
        "BLEU": 13.94974229366959
    },
    "Latency": {
        "AL": 1751.8031870037803,
        "AL_CA": 2338.5911762796536,
        "AP": 0.7931395378788959,
        "AP_CA": 0.9405103863210942,
        "DAL": 1987.7811616943081,
        "DAL_CA": 2425.2751560926167
    }
}
```

**Note**: Quality is measured by detokenized BLEU, and latency metrics include Average Proportion (AP), Average Lagging (AL), and Differentiable Average Lagging (DAL).