# Condensed: XLS-R

Summary: This tutorial covers XLS-R, a cross-lingual speech representation model based on wav2vec 2.0, pretrained on 128 languages. It provides implementation knowledge for ASR fine-tuning using fairseq-hydra-train with specific hyperparameter recommendations for different model sizes (300M, 1B, 2B). The tutorial helps with speech recognition tasks, language identification, and speech translation using CoVoST 2. Key features include model checkpoints, distributed training configuration, activation checkpointing for large models, embedding extraction for language identification, and optimization parameters for different benchmarks (Babel, Common Voice, VoxPopuli, MLS).

*This is a condensed version that preserves essential implementation details and context.*

# XLS-R: Cross-lingual Speech Representation Learning

XLS-R is a set of self-supervised cross-lingual speech representation models based on wav2vec 2.0, pretrained on 128 languages and 436K hours of unlabeled speech data.

## Available Models

| Model | Description | Link |
|-------|-------------|------|
| XLS-R 300M | Base model | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt) |
| XLS-R 1B | Larger model | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_960m_1000k.pt) |
| XLS-R 2B | Largest model | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_2B_1000k.pt) |

Models are also available on [Hugging Face](https://huggingface.co/models?other=xls_r).

## ASR Finetuning Implementation

```shell
fairseq-hydra-train \
    distributed_training.distributed_port=$PORT \
    task.data=/path/to/data \
    model.w2v_path=/path/to/model.pt \
    --config-dir /path/to/fairseq-py/examples/wav2vec/xlsr/config \
    --config-name finetune
```

### Key Hyperparameters

| Benchmark | Total Updates |
|-----------|---------------|
| Babel | 26000 |
| Common Voice | 13000 |
| VoxPopuli | 50000 |
| MLS 10h | 20000 |

- For 300M and 1B models: Use `finetune.yaml` with `optimization.lr` in range [2e-5, 3e-4]
- For 2B model:
  - Use `distributed_training.ddp_backend=fully_sharded` (requires [fairscale](https://github.com/facebookresearch/fairscale))
  - Set `model.activation_checkpoint=true`
  - Increase `dataset.max_tokens` to 2560000 (effective batch size: 2560000*24)
  - Use `optimization.lr` in range [3e-6, 3e-5]
  - For Common Voice: tune `model.mask_prob` from {0.30, 0.40}

## Language Identification Inference

```shell
# Extract embeddings
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python3 examples/wav2vec/gen_audio_embedding.py \
    /path/to/data/manifest --path "/path/to/checkpoint.pt" \
    --task audio_classification --batch-size 90 --gen-subset test \
    --infer-manifest /path/to/data/manifest/test.tsv \
    --infer-xtimes 10 --infer-max-sample-size 160000 --output-path /tmp/output.npz

# Calculate accuracy
PYTHONPATH='.' python examples/wav2vec/eval_speaker_clf_task.py \
    --task cls --merge mean_logit --data /tmp/output.npz
```

## Speech Translation Models

Multilingual finetuned models for [CoVoST 2](https://github.com/facebookresearch/covost) are available for various translation directions and model sizes.