# Condensed: Pay Better Attention to Attention: Head Selection in Multilingual and Multi-Domain Sequence Modeling (Gong et al., 2021)

Summary: This tutorial demonstrates implementing attention head selection strategies for multilingual/multi-domain sequence modeling in speech recognition based on Gong et al. (2021). It covers configuring and training models with specialized attention head selection mechanisms using the fairseq framework. Key features include encoder/decoder attention head selection parameters, different selection strategies (subset or group), and task-specific configurations for language or domain-based modeling. The implementation supports customizing the number of attention heads and includes complete training and inference pipelines with model averaging. This code helps build more efficient multilingual/multi-domain ASR systems through optimized attention mechanisms.

*This is a condensed version that preserves essential implementation details and context.*

# Attention Head Selection in Multilingual/Multi-Domain Sequence Modeling

## Implementation Overview

This tutorial demonstrates how to implement attention head selection strategies for multilingual and multi-domain sequence modeling tasks, particularly for speech recognition, as described in [Gong et al. (2021)](https://arxiv.org/pdf/2106.10840.pdf).

## Data Preparation

Prepare datasets following the standard procedures:
- mTEDx data: Follow the [mTEDx example](https://github.com/fairinternal/fairseq-py/blob/0d9c5851e6fac40f9e366b3633ccd615c2901788/examples/speech_to_text/docs/mtedx_example.md)
- CoVoST data: Follow the [CoVoST example](https://github.com/fairinternal/fairseq-py/blob/0d9c5851e6fac40f9e366b3633ccd615c2901788/examples/speech_to_text/docs/covost_example.md)
- EuroParl data: Similar preparation process

## Training Models with Attention Head Selection

### Multilingual ASR Model

```bash
fairseq-train ${data_dir} \
    --user-dir examples/attention_head_selection/src \
    --train-subset "${train_subset}" \
    --valid-subset "${valid_subset}" \
    --config-yaml 'config_asr.yaml' \
    --arch 'head_selection_s2t_transformer_s' \
    --task 'speech_to_text_head_selection' \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --lr-scheduler 'inverse_sqrt' --warmup-updates 10000 \
    --lr 5e-4 \
    --clip-norm 10.0 \
    --max-epoch 400 \
    --max-tokens 32000 \
    --dropout 0.3 \
    --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
    --encoder-attn-head-select \
    --total-encoder-attention-heads 8 \
    --decoder-self-attn-head-select \
    --total-decoder-attention-heads 8 \
    --attn-head-select-strategy ${strategy} \
    --task-type lang
```

### Multi-Domain ASR Model

```bash
fairseq-train ${data_dir} \
    --user-dir examples/attention_head_selection/src \
    --train-subset "${train_subset}" \
    --valid-subset "${valid_subset}" \
    --config-yaml 'config_asr.yaml' \
    --arch head_selection_s2t_transformer_s \
    --task speech_to_text_head_selection \
    # Same parameters as multilingual model
    --task-type domain
```

## Key Configuration Parameters

- `--encoder-attn-head-select` and `--decoder-self-attn-head-select`: Enable attention head selection
- `--total-encoder-attention-heads 8` and `--total-decoder-attention-heads 8`: Set number of attention heads
- `--attn-head-select-strategy ${strategy}`: Selection strategy (`subset` or `group`)
- `--task-type`: Specify `lang` for multilingual or `domain` for multi-domain tasks

## Inference

### Model Averaging

```bash
python scripts/average_checkpoints.py \
  --inputs ${MODEL_DIR} --num-epoch-checkpoints ${last_n} \
  --output "${MODEL_DIR}/${CHECKPOINT_FILENAME}"
```

### Generation

```bash
fairseq-generate ${data_dir} \
    --user-dir examples/attention_head_selection/src \
    --arch 'head_selection_s2t_transformer_s' \
    --task 'speech_to_text_head_selection' \
    --train-subset ${train_subset} \
    --gen-subset ${gen_subset} \
    --path "${MODEL_DIR}/${CHECKPOINT_FILENAME}" \
    --config-yaml 'config_asr.yaml' \
    --prefix-size 1 \
    --max-tokens 40000 --beam 5 \
    --scoring wer --wer-tokenizer 13a \
    --wer-lowercase --wer-remove-punct --remove-bpe
```

## Citation

```bibtex
@article{gong2021pay,
  title={Pay Better Attention to Attention: Head Selection in Multilingual and Multi-Domain Sequence Modeling},
  author={Gong, Hongyu and Tang, Yun and Pino, Juan and Li, Xian},
  journal={arXiv preprint arXiv:2106.10840},
  year={2021}
}
```