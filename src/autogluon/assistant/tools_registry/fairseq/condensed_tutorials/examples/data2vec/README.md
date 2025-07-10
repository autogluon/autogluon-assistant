# Condensed: data2vec 2.0

Summary: This tutorial explains data2vec 2.0, an efficient self-supervised learning framework that processes only unmasked timesteps and uses convolutional decoders with multimasking. It provides implementation code for training and fine-tuning models across three modalities: vision (ViT models), speech (Librispeech/Libri-light), and NLP (Books+Wiki). The tutorial includes complete command-line instructions for pretraining and fine-tuning in each domain, including GLUE tasks for NLP and CTC for speech recognition. Key features include manifest preparation for speech data, hyperparameter recommendations, and integration with the Fairseq framework using Hydra configuration.

*This is a condensed version that preserves essential implementation details and context.*

# data2vec 2.0: Efficient Self-supervised Learning

data2vec 2.0 improves training efficiency of the original data2vec algorithm through:
- Processing only unmasked timesteps through the encoder
- Using convolutional decoder
- Implementing multimasking to amortize teacher model compute overhead

## Pretrained Models

### Vision Models
```
data2vec ViT-B, ViT-L, ViT-H (pretrained and Imagenet-1K finetuned versions available)
```

### Speech Models
```
data2vec Base (Librispeech, pretrained and 960h finetuned)
data2vec Large (Libri-light, pretrained and 960h finetuned)
```

### NLP Models
```
data2vec Base (Books + Wiki, pretrained)
```

## Training Commands

### Vision Pretraining
```shell
python fairseq_cli/hydra_train.py -m --config-dir examples/data2vec/config/v2 \
--config-name base_images_only_task task.data=/path/to/dir
```

### Vision Finetuning
```shell
python fairseq_cli/hydra_train.py -m --config-dir examples/data2vec/config/vision/finetuning \
--config-name mae_imagenet_clean task.data=/path/to/dir model.model_path=/path/to/pretrained/model
```

### Speech Pretraining
```shell
python fairseq_cli/hydra_train.py -m --config-dir examples/data2vec/config/v2 \
--config-name base_audio_only_task task.data=/path/to/manifests
```

### Speech Finetuning
```shell
python fairseq_cli/hydra_train.py -m --config-dir examples/wav2vec/config/finetuning \
--config-name vox_10h task.data=/path/to/manifests model.w2v_path=/path/to/pretrained/model \
common.user_dir=examples/data2vec
```

### NLP Pretraining
```shell
python fairseq_cli/hydra_train.py -m --config-dir examples/data2vec/config/v2 \
--config-name base_text_only_task task.data=/path/to/file
```

### NLP GLUE Finetuning
```shell
task=cola  # choose from [cola|qnli|mrpc|rte|sst_2|mnli|qqp|sts_b]
lr=1e-5    # sweep [1e-5|2e-5|4e-5|6e-5] for each task
python fairseq_cli/hydra_train.py -m --config-dir examples/data2vec/config/v2/text_finetuning \
--config-name $task task.data=/path/to/file model.model_path=/path/to/pretrained/model \
"optimization.lr=[${lr}]"
```

## Original data2vec

For speech model training:

1. Prepare manifest:
```shell
pip install soundfile
python examples/wav2vec/wav2vec_manifest.py /path/to/waves --dest /manifest/path --ext $ext --valid-percent $valid
```

2. Train base model:
```shell
python fairseq_cli/hydra_train.py -m --config-dir examples/data2vec/config/audio/pretraining \
--config-name base_librispeech task.data=/path/to/manifests common.user_dir=examples/data2vec
```

3. Fine-tune with CTC:
```shell
fairseq-hydra-train \
    distributed_training.distributed_port=$PORT \
    task.data=/path/to/data \
    model.w2v_path=/path/to/model.pt \
    --config-dir /path/to/fairseq-py/examples/wav2vec/config/finetuning \
    --config-name base_100h common.user_dir=examples/data2vec
```