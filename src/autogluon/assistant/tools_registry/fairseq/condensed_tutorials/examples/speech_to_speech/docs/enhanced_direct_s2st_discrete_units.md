# Condensed: Speech to speech translation (S2ST)

Summary: This tutorial implements speech-to-speech translation (S2ST) using discrete units, specifically the speech-to-unit translation (S2UT) approach. It covers implementation of models that combine wav2vec 2.0 and unit mBART components for translating speech between languages. Key features include: data preparation with discrete speech units, model training with various fine-tuning options (including LNA-E and LNA-D partial fine-tuning), inference pipeline to generate translated speech, and evaluation methods using ASR and BLEU. The tutorial provides complete code examples for training, inference, and waveform generation, making it valuable for implementing direct speech-to-speech translation systems with pretrained components.

*This is a condensed version that preserves essential implementation details and context.*

# Speech to Speech Translation (S2ST) Implementation

This guide covers the implementation of speech-to-unit translation (S2UT) as proposed in [Enhanced Direct Speech-to-Speech Translation Using Self-supervised Pre-training and Data Augmentation](https://arxiv.org/abs/2204.02967).

## Pretrained Models

### Key Components
- **Unit extraction**: Multilingual HuBERT model from Textless S2ST
- **Wav2vec 2.0**: Various pretrained models for Spanish and English
- **Unit mBART**: 1000-unit model trained on Voxpopuli English and Spanish unlabelled speech

## Data Preparation

1. Format data in S2UT format following the "Direct S2ST with Discrete Units" guide
2. Use 1000 units from the 11th layer of the multilingual HuBERT model
3. Update TSV headers:
```bash
var="id\taudio\tn_frames\ttgt_text\ttgt_n_frames"
sed -i "1s/.*/$var/" ${SPLIT}.tsv
```

## Training

### Speech-to-Unit Translation (S2UT)

```bash
fairseq-train $DATA_ROOT \
  --config-yaml config.yaml \
  --task speech_to_text --arch xm_transformer \
  --criterion l --label-smoothing 0.2 \
  --share-decoder-input-output-embed --adaptor-n-layers 1 --normalize \
  --dropout 0.1 --attention-dropout 0.1 --relu-dropout 0.1 \
  --train-subset train --valid-subset dev \
  --load-pretrained-decoder-from ${unit_mBART} --w2v-path ${wav2vec2.0} \
  --mask-prob 0.3 --mask-channel-length 32 --mask-channel-prob 0.25 \
  --save-dir ${MODEL_DIR} --checkpoint-activations --encoder-proj \
  --lr 0.0005 --lr-scheduler inverse_sqrt \
  --warmup-init-lr 1e-7 --warmup-updates 10000 \
  --optimizer adam --adam-betas "(0.9,0.98)" --clip-norm 10.0 \
  --max-update 20000 --max-tokens 4000 --max-tokens-valid 4000 \
  --max-source-positions 4000 --max-target-positions 4000 \
  --update-freq 120 --seed 1 --fp16 --num-workers 1
```

**Important training options:**
- Adjust `--update-freq` based on available GPUs (above simulates 120 GPUs)
- For LNA-E partial finetuning: `--finetune-w2v-params layer_norm,self_attn`
- For LNA-D partial finetuning: `--finetune-decoder-params encoder_attn,layer_norm,self_attn`
- To freeze encoder for k updates: `--freeze-finetune-updates ${K}`

## Inference

### Speech-to-Unit Translation

1. Generate unit sequences:
```bash
fairseq-generate $DATA_ROOT \
  --config-yaml config.yaml \
  --task speech_to_text \
  --path $MODEL_DIR/checkpoint_best.pt --gen-subset $GEN_SUBSET \
  --max-tokens 10000 --max-source-positions 10000 --max-target-positions 10000 \
  --beam 10 --max-len-a 1 --max-len-b 200 \
  --results-path ${RESULTS_PATH}
```

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

## Evaluation

Evaluation process:
1. Apply ASR on speech output
2. Compute BLEU score between ASR text and references using sacreBLEU

**ASR Models:**
- English: Wav2Vec 2.0 Large (LV-60) + Self Training / 960 hours
- Spanish: Wav2Vec2-Large-XLSR-53-Spanish from Hugging Face

## Finetuned Model Checkpoints

Multiple checkpoints are available for:
- S2UT systems without pre-training
- S2UT systems with model pre-training (w2v2-L, w2v2-L + mBART with various configurations)
- S2UT systems with model pre-training and data augmentation

**Note:** When using models that use `speech_to_text_sharded` task, override the task to `speech_to_text`.