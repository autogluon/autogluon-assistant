# Condensed: [[Back]](..)

Summary: This tutorial implements a unified speech-text pre-training framework for speech translation and recognition using Fairseq. It covers techniques for joint speech-text representation learning through multi-task pre-training with phoneme alignment, CTC-based alternatives, and fine-tuning procedures. The implementation helps with building models that can process both speech and text inputs for ASR and speech translation tasks. Key features include: multi-modal pre-training with speech and text data, forced alignment integration, model architecture for joint speech-text processing, fine-tuning procedures for downstream tasks, and evaluation methods for LibriSpeech ASR and MuST-C speech translation with complete command-line examples.

*This is a condensed version that preserves essential implementation details and context.*

# Unified Speech-Text Pre-training for Speech Translation and Recognition

## Data Preparation

### Required Data Types
1. **Text to text task (T2T)**: Source data as phoneme token sequences, target data as subword tokens via SentencePiece
2. **Self-supervised speech learning task (SSL)**: Prepared as wav2vec 2.0
3. **Speech to phoneme classification task (S2P)**: TSV file with 5 fields: "id", "audio", "n_frames", "tgt_text", "align"
   - Phoneme-level forced alignment can be obtained via kaldi or MFA
   - Segmentation normalized to 0~1 for each utterance
4. **Speech to text task (S2T)**: Follows EN_DE Joint training preparation

## Implementation

### Pre-training Command
```bash
python train.py $T2T_DATA \
    --save-dir $SAVE_PRE_PATH --user-dir examples/speech_text_joint_to_text --task speech_text_joint_denoising \
    --criterion speech_text_pretrain_cross_entropy --optimizer adam --weight-decay 0.01 \
    --config-yaml config_s2p.yaml --config-s2s-yaml config.yaml --ddp-backend no_c10d \
    --lang-pairs pho-wrd --num-workers 4 --log-interval 500 \
    --save-interval-updates 5000 --keep-interval-updates 1 --no-emb-update-unsup \
    --report-accuracy --lr 0.001 --end-learning-rate 1e-06 \
    --lr-scheduler polynomial_decay --warmup-updates 10000 --total-num-update 800000 \
    --update-freq 6 --validate-interval-updates 10000 --train-subset train \
    --valid-subset valid,valid_sup_speech,valid_sup_speech_s2s,valid_unsup_speech \
    --dataset-impl mmap \
    --sup-speech-data $S2P_DATA_PATH --sup-speech-train-subset train_960.ali \
    --sup-speech-valid-subset dev-clean.ali --sup-speech-s2s-data $S2T_DATA_PATH \
    --sup-speech-s2s-train-subset train --sup-speech-s2s-valid-subset dev-clean \
    --unsup-speech-train-data $SSL_DATA_PATH/train.tsv --unsup-speech-valid-data $SSL_DATA_PATH/valid.tsv \
    --batch-size 200 --batch-size-valid 150 --max-source-positions 1024 \
    --max-target-positions 1024 --max-text-tokens 3072 --max-speech-positions 600000 \
    --max-sample-size 750000 --min-sample-size 64000 --max-speech-tokens 750000 \
    --max-tokens-valid 750000 --skip-invalid-size-inputs-valid-test \
    --unsupervised-speech-sample-ratio 3.0 --supervised-speech-sample-ratio 5 \
    --supervised-speech-s2s-sample-ratio 5 --text-sample-ratio 1.0 \
    --mask 0.3 --mask-random 0.1 --mask-length span-poisson \
    --speech-sup-mask-prob 0.3 --speech-unsup-mask-prob 0.7 --use-mask-whole-words \
    --arch speech_text_pretrain_bart_base_stack --no-scale-feature --activation-fn gelu \
    --speech-extractor-mode default --stacked-encoder all \
    --encoder-normalize-before --decoder-normalize-before \
    --encoder-learned-pos --decoder-learned-pos --dropout 0.1 \
    --load-pretrained-mbart-encoder-from $BART --load-pretrained-mbart-decoder-from $BART
```

### CTC-based Pre-training Alternative
For pre-training without forced alignment data:
- **Add options**: `--use-sup-speech-ctc --criterion speech_text_pretrain_compound`
- **Remove options**: `--same-data-update --criterion speech_text_pretrain_cross_entropy`
- Note: CTC-based pre-training performs worse than forced alignment-based setting

### Fine-tuning Command
```bash
python train.py $S2T_DATA_PATH \
    --save-dir $SAVE_FT_PATH --num-workers 8 --task speech_text_joint_to_text \
    --arch dualinputs2twavtransformer_base_stack \
    --user-dir examples/speech_text_joint_to_text --max-update 100000 \
    --optimizer adam --lr-scheduler inverse_sqrt --lr 0.0003 --update-freq 3 \
    --clip-norm 10.0 \
    --criterion guided_label_smoothed_cross_entropy_with_accuracy --guide-alpha 0.8 \
    --label-smoothing 0.1 --warmup-updates 20000 --attentive-cost-regularization 0.02 \
    --enc-grad-mult 2.0 --max-tokens 800000 --max-source-positions 800000 \
    --max-tokens-text 10000 --max-positions-text 1024 --max-target-positions 1024 \
    --no-scale-feature --activation-fn gelu \
    --load-pretrained-speech-text-encoder $SAVE_PRE_PATH/checkpoint_last.pt \
    --load-pretrained-speech-text-decoder $SAVE_PRE_PATH/checkpoint_last.pt \
    --encoder-normalize-before --decoder-normalize-before --speech-extractor-mode default \
    --speech-mask-channel-length 64 --speech-mask-channel-prob 0.5 \
    --speech-mask-length 10 --speech-mask-prob 0.65 --text-sample-ratio 0.25 \
    --mask-text-ratio 0.3 --mask-text-type random --parallel-text-data text_bin \
    --text-input-cost-ratio 0.5 --langpairs pho-wrd --update-mix-data \
    --log-format json --max-tokens-valid 800000 --ddp-backend no_c10d --log-interval 500 \
    --config-yaml config.yaml --skip-invalid-size-inputs-valid-test --keep-last-epochs 50 \
    --layernorm-embedding --encoder-learned-pos --decoder-learned-pos
```

# Evaluation and Results

## Model Evaluation

The final model is evaluated using model averaging of the last 10 epochs from fine-tuning:

```python
python ./fairseq_cli/generate.py \
    $S2T_DATA_PATH \
    --task speech_text_joint_to_text \
    --max-tokens 800000 \
    --max-source-positions 800000 \
    --nbest 1 \
    --results-path $RESULTS_LOG \
    --batch-size 512 \
    --path $FINAL_MODEL \
    --gen-subset $SUBSET \
    --config-yaml config.yaml \
    --scoring wer \
    --beam 10 --lenpen 1.0 \
    --user-dir examples/speech_text_joint_to_text \
    --load-speech-only \
    --model-overrides {'load_pretrained_speech_text_decoder':'','load_pretrained_speech_text_encoder':''}
```

## LibriSpeech Results

| Dataset | dev-clean | dev-other | test-clean | test-other |
|---------|-----------|-----------|------------|------------|
| WER     | 2.0       | 4.4       | 2.1        | 4.6        |

## Pre-trained Model Resources

- **Configuration files**: `config_s2p.yaml`, `config.yaml`
- **Tokenization**: `spm.model`, `src_dict.txt`, `tgt_dict.txt`
- **Models**: BART (trained on LibriSpeech text), Joint Pre-trained model, Fine-tuned model

## MuST-C Implementation

### Pre-training (EN-DE example)
```python
python train.py $TXT_DATA \
    --save-dir $SAVE_PRE_PATH \
    --user-dir examples/speech_text_joint_to_text \
    --task speech_text_joint_denoising \
    --criterion speech_text_pretrain_cross_entropy \
    --optimizer adam --weight-decay 0.01 \
    --config-yaml config_s2p.yaml --config-s2s-yaml config.yaml \
    --lang-pairs-bitext en-fr \
    --no-emb-update-unsup --use-decoder-output-proj \
    --lr 0.001 --end-learning-rate 1e-06 \
    --lr-scheduler polynomial_decay \
    --warmup-updates 10000 --total-num-update 800000 \
    --update-freq 8 \
    --sup-speech-data $S2P_DATA_PATH \
    --sup-speech-s2s-data $S2T_DATA_PATH \
    --unsup-speech-train-data $SSL_DATA_PATH/train.tsv \
    --max-speech-positions 600000 --max-sample-size 600000 \
    --unsupervised-speech-sample-ratio 1.2 \
    --supervised-speech-sample-ratio 10 \
    --supervised-speech-s2s-sample-ratio 10 \
    --bitext-sample-ratio 0.5 \
    --mask 0.3 --mask-random 0.1 \
    --speech-sup-mask-prob 0.3 --speech-unsup-mask-prob 0.7 \
    --arch speech_text_pretrain_bart_base_stack \
    --load-pretrained-mbart-encoder-from $EN_FR_NMT \
    --load-pretrained-mbart-decoder-from $EN_FR_NMT
```

### Fine-tuning
```python
python train.py $S2T_DATA_PATH \
    --save-dir $SAVE_FT_PATH \
    --task speech_text_joint_to_text \
    --arch dualinputs2twavtransformer_base_stack \
    --user-dir examples/speech_text_joint_to_text \
    --max-epoch 25 --update-mix-data \
    --optimizer adam --lr-scheduler inverse_sqrt --lr 0.0003 \
    --update-freq 4 --clip-norm 10.0 --warmup-updates 20000 \
    --criterion guided_label_smoothed_cross_entropy_with_accuracy \
    --guide-alpha 0.8 --attentive-cost-regularization 0.02 \
    --enc-grad-mult 2.0 --label-smoothing 0.1 \
    --max-tokens 800000 --max-source-positions 800000 \
    --load-pretrained-speech-text-encoder $SAVE_PRE_PATH/checkpoint_last.pt \
    --load-pretrained-speech-text-decoder $SAVE_PRE_PATH/checkpoint_last.pt \
    --speech-mask-channel-length 64 --speech-mask-channel-prob 0.5 \
    --speech-mask-length 10 --speech-mask-prob 0.65 \
    --text-sample-ratio 0.05 --mask-text-ratio 0.3 \
    --mask-text-type random --parallel-text-data data-bin-wt \
    --text-input-cost-ratio 0.5 --langpairs en-fr
```

### MuST-C Evaluation
```python
python fairseq_cli/generate.py \
    $S2T_DATA_PATH \
    --task speech_text_joint_to_text \
    --nbest 1 --max-tokens 800000 --max-source-positions 800000 \
    --results-path $RESULTS_LOG --batch-size 512 \
    --path $FINAL_MODEL --gen-subset $SUBSET \
    --config-yaml config.yaml --scoring sacrebleu \
    --beam 10 --lenpen 1.0 \
    --user-dir examples/speech_text_joint_to_text \
    --load-speech-only \
    --model-overrides {'load_pretrained_speech_text_decoder':'','load_pretrained_speech_text_encoder':''}
```

Key differences from LibriSpeech implementation:
- MuST-C speech data replaces LibriSpeech data
- WMT parallel text data replaces LibriSpeech text data
- Evaluation uses sacrebleu scoring instead of WER

# Results and Models

## Performance Metrics
| Language Pair | BLEU Score |
|---------------|------------|
| English-French | 39.7 |
| English-Spanish | 33.2 |
| English-German | 29.2 |

## Pre-trained Models and Resources

The following resources are available for each language pair (German, Spanish, and French):

- Configuration files (`config.yaml`)
- Dictionary files (`src_dict.txt`, `tgt_dict.txt`)
- SentencePiece models (`spm.model`)
- Pre-trained NMT models (`nmt.pt`)
- Pre-trained speech-to-text models (`checkpoint_pretraing.pt`)
- Fine-tuned models (`checkpoint_finetune_ave10.pt`)

### German (DE) Resources
```
https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/acl2022/must_c/de/
```

### Spanish (ES) Resources
```
https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/acl2022/must_c/es/
```

### French (FR) Resources
```
https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/acl2022/must_c/fr/
```


...(truncated)