# Condensed: Direct speech-to-speech translation with discrete units

Summary: This tutorial implements direct speech-to-speech translation using two approaches: Speech-to-Unit Translation (S2UT) and Speech-to-Spectrogram Translation (S2SPECT) from the paper "Direct speech-to-speech translation with discrete units." It covers data preparation workflows for converting audio into discrete units or spectrograms, model training configurations for transformer-based architectures, and inference pipelines including unit-to-waveform conversion. Key functionalities include multitask learning support, HiFi-GAN vocoder integration, and evaluation methods. The implementation helps with end-to-end speech translation tasks without requiring intermediate text representations, supporting both reduced and stacked unit approaches with detailed command-line instructions.

*This is a condensed version that preserves essential implementation details and context.*

# Direct Speech-to-Speech Translation with Discrete Units

This implementation covers speech-to-unit translation (S2UT) from the paper "Direct speech-to-speech translation with discrete units" and transformer-based speech-to-spectrogram translation (S2SPECT).

## Pretrained Models

### Unit-based HiFi-GAN Vocoder
- **HuBERT Base, Librispeech (layer 6)**: 100 unit size, trained on LJSpeech
  - [Checkpoint](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/hubert_base_100_lj/g_00500000)
  - [Config](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/hubert_base_100_lj/config.json)

## Data Preparation

### Target Speech
1. Prepare two folders with audio files: `$SRC_AUDIO` and `$TGT_AUDIO` with `${SPLIT}/${SAMPLE_ID}.wav` structure
   - S2UT: target audio should be 16kHz
   - S2SPECT: target audio recommended at 22.05kHz

2. For S2UT, prepare discrete units using [speech2unit](https://github.com/pytorch/fairseq/tree/main/examples/textless_nlp/gslm/speech2unit)
   - Set output as `${TGT_AUDIO}/${SPLIT}.txt`
   - The paper uses 100 units from layer 6 of HuBERT Base

### Formatting Data

**S2UT Data Preparation**
```bash
python examples/speech_to_speech/preprocessing/prep_s2ut_data.py \
  --source-dir $SRC_AUDIO --target-dir $TGT_AUDIO --data-split $SPLIT1 $SPLIT2 \
  --output-root $DATA_ROOT --reduce-unit \
  --vocoder-checkpoint $VOCODER_CKPT --vocoder-cfg $VOCODER_CFG
```

**S2SPECT Data Preparation**
```bash
python examples/speech_to_speech/preprocessing/prep_s2spect_data.py \
  --source-dir $SRC_AUDIO --target-dir $TGT_AUDIO --data-split $SPLIT1 $SPLIT2 \
  --output-root $DATA_ROOT
```

**Multitask Data**
1. For each task, prepare `${DATA_ROOT}/${TASK_NAME}/${SPLIT}.tsv` with tab-separated columns:
   ```
   id  tgt_text
   sample_id_0 token1 token2 token3 ...
   ```

2. Create dictionary file `${DATA_ROOT}/${TASK_NAME}/dict.txt`

3. Create `config_multitask.yaml` (example for S2UT reduced with Fisher):
   ```yaml
   source_letter:
     decoder_type: transformer
     dict: ${DATA_ROOT}/source_letter/dict.txt
     data: ${DATA_ROOT}/source_letter
     encoder_layer: 6
     loss_weight: 8.0
   target_letter:
     decoder_type: transformer
     dict: ${DATA_ROOT}/target_letter/dict.txt
     data: ${DATA_ROOT}/target_letter
     encoder_layer: 8
     loss_weight: 8.0
   decoder_target_ctc:
     decoder_type: ctc
     dict: ${DATA_ROOT}/decoder_target_ctc/dict.txt
     data: ${DATA_ROOT}/decoder_target_ctc
     decoder_layer: 3
     loss_weight: 1.6
   ```

## Training

### Speech-to-Unit Translation (S2UT)

```bash
fairseq-train $DATA_ROOT \
  --config-yaml config.yaml --multitask-config-yaml config_multitask.yaml \
  --task speech_to_speech --target-is-code --target-code-size 100 --vocoder code_hifigan \
  --criterion speech_to_unit --label-smoothing 0.2 \
  --arch s2ut_transformer_fisher --share-decoder-input-output-embed \
  --dropout 0.1 --attention-dropout 0.1 --relu-dropout 0.1 \
  --train-subset train --valid-subset dev \
  --save-dir ${MODEL_DIR} \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-7 --warmup-updates 10000 \
  --optimizer adam --adam-betas "(0.9,0.98)" --clip-norm 10.0 \
  --max-update 400000 --max-tokens 20000 --max-target-positions 3000 --update-freq 4 \
  --seed 1 --fp16 --num-workers 8
```

**Important notes:**
- Adjust `--update-freq` based on your GPU count
- For S2UT _stacked_ system, set `--n-frames-per-step 5` and use data prepared without `--reduce-unit`
- Optional: Track MCD loss during training with `--eval-inference --eval-args '{"beam": 1, "max_len_a": 1}' --best-checkpoint-metric mcd_loss`

### Speech-to-Spectrogram Translation (S2SPECT)

```bash
fairseq-train $DATA_ROOT \
  --config-yaml config.yaml --multitask-config-yaml config_multitask.yaml \
  --task speech_to_speech --n-frames-per-step 5 \
  --criterion speech_to_spectrogram \
  --arch s2spect_transformer_fisher --decoder-normalize-before \
  --dropout 0.1 --attention-dropout 0.1 --relu-dropout 0.1 \
  --train-subset train --valid-subset dev \
  --save-dir ${MODEL_DIR} \
  --eval-inference --best-checkpoint-metric mcd_loss \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-7 --warmup-updates 10000 \
  --optimizer adam --adam-betas "(0.9,0.98)" --clip-norm 10.0 --weight-decay 1e-6 \
  --max-update 400000 --max-tokens 80000 --max-tokens-valid 30000 --required-batch-size-multiple 1 \
  --max-target-positions 3000 --update-freq 16 \
  --seed 1 --fp16 --num-workers 8
```

## Inference

### Speech-to-Unit Translation (S2UT)

1. Generate unit sequences:
   ```bash
   fairseq-generate $DATA_ROOT \
     --config-yaml config.yaml --multitask-config-yaml config_multitask.yaml \
     --task speech_to_speech --target-is-code --target-code-size 100 --vocoder code_hifigan \
     --path $MODEL_DIR/checkpoint_best.pt --gen-subset $GEN_SUBSET \
     --max-tokens 50000 \
     --beam 10 --max-len-a 1 \
     --results-path ${RESULTS_PATH}
   ```
   - For S2UT _stacked_ models, use `--beam 1 --n-frames-per-step $r`

2. Convert units to waveform:
   ```bash
   grep "^D\-" ${RESULTS_PATH}/generate-${GEN_SUBSET}.txt | \
     sed 's/^D-//ig' | sort -nk1 | cut -f3 \
     > ${RESULTS_PATH}/generate-${GEN_SUBSET}.unit

   python examples/speech_to_speech/generate_waveform_from_code.py \
     --in-code-file ${RESULTS_PATH}/generate-${GEN_SUBSET}.unit \
     --vocoder $VOCODER_CKPT --vocoder-cfg $VOCODER_CFG \
     --results-path ${RESULTS_PATH} --dur-prediction
   ```
   - Use `--dur-prediction` for S2UT _reduced_ models

### Speech-to-Spectrogram Translation (S2SPECT)

```bash
python examples/speech_synthesis/generate_waveform.py $DATA_ROOT \
  --config-yaml config.yaml --multitask-config-yaml config_multitask.yaml \
  --task speech_to_speech --n-frames-per-step 5 \
  --path $MODEL_DIR/checkpoint_best.pt --gen-subset $GEN_SUBSET \
  --max-tokens 50000 \
  --results-path ${RESULTS_PATH} --dump-waveforms --output-sample-rate 16000
```

## Evaluation

For speech translation evaluation:
1. Apply ASR on speech output
2. Compute BLEU score between ASR text and references using sacreBLEU

**English ASR**:
- Use [Wav2Vec 2.0 Large (LV-60) + Self Training](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_vox_960h_pl.pt)
- Text normalization: Use [keithito/tacotron](https://github.com/keithito/tacotron) text cleaner