# Condensed: [[Back]](..)

Summary: This tutorial demonstrates implementing joint speech-text training for speech translation on MuST-C English-German dataset using Fairseq. It covers techniques for phoneme-based text representation, parallel data preparation, and dual-input transformer architecture. Key functionalities include: data preprocessing with grapheme-to-phoneme conversion, integrating parallel text data from WMT, training with guided label smoothing and attentive cost regularization, and leveraging pretrained encoders/decoders. The tutorial provides complete training scripts for both baseline and enhanced joint training approaches, along with evaluation methods and benchmark results across multiple language pairs (En-De, En-Es, En-Fr).

*This is a condensed version that preserves essential implementation details and context.*

# Joint Speech Text Training for MuST-C English to German Speech Translation

## Data Preparation

### Required Files
- Download [spm.model](https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/must_c/en_de/spm.model), [dict.txt](https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/must_c/en_de/dict.txt), and [config.yaml](https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/must_c/en_de/config.yaml)

### MuST-C Dataset Setup
1. Follow [S2T example](https://github.com/pytorch/fairseq/blob/main/examples/speech_to_text/docs/mustc_example.md) for initial data preparation
2. Convert source text to phoneme representation:
```bash
python examples/speech_text_joint_to_text/scripts/g2p_encode.py \
    --lower-case --do-filter --use-word-start --no-punc \
    --reserve-word examples/speech_text_joint_to_text/configs/mustc_noise.list \
    --data-path ${must_c_en_de_src_text} \
    --out-path ${must_c_en_de_src_text_pho}
```
3. Replace "src_text" column in tsv with phoneme representation
4. Prepare phoneme dictionary as src_dict.txt

### WMT Text Data
1. Download WMT data using prepare-wmt14en2de.sh
2. Convert English text to phoneme representation
3. Generate binary parallel files with fairseq-preprocess

## Training

### Pretrained Models
- Download [pretrain_encoder](https://dl.fbaipublicfiles.com/fairseq/s2t/mustc_joint_asr_transformer_m.pt)
- Download [pretrain_nmt](https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/must_c/en_de/checkpoint_mt.pt)

### Training Scripts

#### Baseline Joint Training
```bash
python train.py ${MANIFEST_ROOT} \
    --save-dir ${save_dir} \
    --num-workers 8 \
    --task speech_text_joint_to_text \
    --arch dualinputs2ttransformer_s \
    --user-dir examples/speech_text_joint_to_text \
    --max-epoch 100 --update-mix-data \
    --optimizer adam --lr-scheduler inverse_sqrt \
    --lr 0.001 --update-freq 4 --clip-norm 10.0 \
    --criterion guided_label_smoothed_cross_entropy_with_accuracy \
    --label-smoothing 0.1 --max-tokens 10000 --max-tokens-text 10000 \
    --max-positions-text 400 --seed 2 --speech-encoder-layers 12 \
    --text-encoder-layers 6 --encoder-shared-layers 6 --decoder-layers 6 \
    --dropout 0.1 --warmup-updates 20000  \
    --text-sample-ratio 0.25 --parallel-text-data ${parallel_text_data} \
    --text-input-cost-ratio 0.5 --enc-grad-mult 2.0 --add-speech-eos \
    --log-format json --langpairs en-de --noise-token '▁NOISE' \
    --mask-text-ratio 0.0 --max-tokens-valid 20000 --ddp-backend no_c10d \
    --log-interval 100 --data-buffer-size 50 --config-yaml config.yaml \
    --keep-last-epochs 10
```

#### Enhanced Joint Training
```bash
python train.py ${MANIFEST_ROOT} \
    --save-dir ${save_dir} \
    --num-workers 8 \
    --task speech_text_joint_to_text \
    --arch dualinputs2ttransformer_m \
    --user-dir examples/speech_text_joint_to_text \
    --max-epoch 100 --update-mix-data \
    --optimizer adam --lr-scheduler inverse_sqrt \
    --lr 0.002 --update-freq 4 --clip-norm 10.0 \
    --criterion guided_label_smoothed_cross_entropy_with_accuracy \
    --guide-alpha 0.8 --disable-text-guide-update-num 5000 \
    --label-smoothing 0.1 --max-tokens 10000 --max-tokens-text 10000 \
    --max-positions-text 400 --seed 2 --speech-encoder-layers 12 \
    --text-encoder-layers 6 --encoder-shared-layers 6 --decoder-layers 6 \
    --dropout 0.1 --warmup-updates 20000 --attentive-cost-regularization 0.02 \
    --text-sample-ratio 0.25 --parallel-text-data ${parallel_text_data} \
    --text-input-cost-ratio 0.5 --enc-grad-mult 2.0 --add-speech-eos \
    --load-pretrain-speech-encoder ${pretrain_encoder} \
    --load-pretrain-decoder ${pretrain_nmt} \
    --load-pretrain-text-encoder-last ${pretrain_nmt} \
    --keep-last-epochs 10
```

## Evaluation
```bash
python ./fairseq_cli/generate.py \
    ${MANIFEST_ROOT} \
    --task speech_text_joint_to_text \
    --max-tokens 25000 \
    --nbest 1 \
    --results-path ${infer_results} \
    --batch-size 512 \
    --path ${model} \
    --gen-subset tst-COMMON_st \
    --config-yaml config.yaml \
    --scoring sacrebleu \
    --beam 5 --lenpen 1.0 \
    --user-dir examples/speech_text_joint_to_text \
    --load-speech-only
```

## Results (Enhanced Joint Training)
| Direction | En-De | En-Es | En-Fr |
|-----------|-------|-------|-------|
| BLEU      | 27.4  | 31.2  | 37.6  |
| Checkpoint| [link](https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/must_c/en_de/checkpoint_ave_10.pt) | [link](https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/must_c/en_es/checkpoint_ave_10.pt) | [link](https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/must_c/en_fr/checkpoint_ave_10.pt) |