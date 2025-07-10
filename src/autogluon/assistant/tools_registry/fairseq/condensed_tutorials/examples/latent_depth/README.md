# Condensed: Deep Transformers with Latent Depth (Li et al., 2020)

Summary: This tutorial demonstrates implementing deep transformers with latent depth in Fairseq, a technique that automatically learns optimal layer selection through posterior distributions. It covers how to train multilingual machine translation models where different language pairs can utilize different subsets of a shared transformer network. Key functionalities include configuring latent depth in decoders, controlling sparsity regularization, annealing updates, and layer sharing weights. The implementation provides command-line arguments for both training and inference, specifically designed for multilingual translation tasks with the TED8 dataset. This approach enables more efficient transformer models by dynamically determining which layers are necessary for each language pair.

*This is a condensed version that preserves essential implementation details and context.*

# Deep Transformers with Latent Depth

## Implementation Overview

This framework automatically learns which layers to use by learning posterior distributions of layer selection, enabling training of a shared Transformer network for multilingual machine translation with different layer selection posteriors for each language pair.

## Training with Latent Depth

Example for training a one-to-many (O2M) multilingual model with latent depth in the decoder:

```bash
lang_pairs_str="eng-aze,eng-bel,eng-ces,eng-glg,eng-por,eng-rus,eng-slk,eng-tur"
databin_dir=<path to binarized data>

fairseq-train ${databin_dir} \
  --user-dir examples/latent_depth/latent_depth_src \
  --lang-pairs "${lang_pairs_str}" \
  --arch multilingual_transformer_iwslt_de_en \
  --task multilingual_translation_latent_depth \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --share-encoders \
  --share-decoders \
  --decoder-langtok \
  --share-decoder-input-output-embed \
  --dropout 0.3 --attention-dropout 0.3 \
  --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
  --lr-scheduler inverse_sqrt --stop-min-lr 1e-9 --warmup-init-lr 1e-7 --warmup-updates 8000 \
  --max-tokens 4096 --update-freq 1  \
  --lr 0.0015 \
  --clip-norm 1.0 \
  --seed 2 \
  --ddp-backend=legacy_ddp \
  --encoder-layers 12 \
  --decoder-layers 24 \
  --decoder-latent-layer \
  --sparsity-weight 0.1 \
  --anneal-updates 5000 \
  --soft-update 500  \
  --target-layers 12 \
  --share-weight 0.1
```

## Key Configuration Parameters

- `--decoder-latent-layer`: Enables latent depth in decoder
- `--sparsity-weight 0.1`: Controls sparsity regularization
- `--anneal-updates 5000`: Number of updates for annealing
- `--soft-update 500`: Controls soft update frequency
- `--target-layers 12`: Target number of layers to use
- `--share-weight 0.1`: Weight for layer sharing

## Inference Command

```bash
fairseq-generate ${databin_dir} \
  --path ${model_path} \
  --task multilingual_translation_latent_depth \
  --decoder-latent-layer \
  --lang-pairs "${lang_pairs_str}" \
  -s ${src_lang} -t ${tgt_lang} \
  --gen-subset $gen_data \
  --scoring sacrebleu \
  --remove-bpe 'sentencepiece' \
  --lenpen 1.0 \
  --beam 5  \
  --decoder-langtok \
  --max-tokens 4096
```

The implementation uses the TED8 dataset, which can be preprocessed using scripts from the multiDDS repository.