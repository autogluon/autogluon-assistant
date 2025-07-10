# Condensed: Neural Machine Translation with Byte-Level Subwords

Summary: This tutorial demonstrates implementing neural machine translation using byte-level byte-pair encoding (BBPE) with the IWSLT 2017 Fr-En dataset. It covers how to build a Transformer model with Bi-GRU embedding contextualization, comparing various vocabulary types (bytes, chars, BPE, BBPE) with different sizes. The implementation includes complete code for data preparation, model training with fairseq, and generation/inference with both batch and interactive translation modes. Key features include byte-level decoding for BBPE, Moses tokenization integration, and performance benchmarks showing how different vocabulary strategies affect BLEU scores.

*This is a condensed version that preserves essential implementation details and context.*

# Neural Machine Translation with Byte-Level Subwords

This implementation demonstrates byte-level byte-pair encoding (BBPE) for neural machine translation using IWSLT 2017 Fr-En data.

## Data Preparation
```bash
bash ./get_data.sh
```

## Model Training
Train a Transformer model with Bi-GRU embedding contextualization:

```bash
# Choose vocabulary type (bbpe2048 is the default example)
VOCAB=bbpe2048  # Options: bytes, chars, bbpe2048, bpe2048, bbpe4096, bpe4096, bpe16384

fairseq-train "data/bin_${VOCAB}" --task translation --user-dir examples/byte_level_bpe/gru_transformer \
    --arch gru_transformer --encoder-layers 2 --decoder-layers 2 --dropout 0.3 --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --log-format 'simple' --log-interval 100 --save-dir "checkpoints/${VOCAB}" \
    --batch-size 100 --max-update 100000 --update-freq 2
```

## Generation
For BBPE, a byte-level decoder is required to convert representations back to characters:

```bash
# Set BPE configuration based on vocabulary type
BPE=--bpe byte_bpe --sentencepiece-model-path data/spm_bbpe2048.model  # For bbpe2048

# Generate translations
fairseq-generate "data/bin_${VOCAB}" --task translation --user-dir examples/byte_level_bpe/gru_transformer \
    --source-lang fr --gen-subset test --sacrebleu --path "checkpoints/${VOCAB}/checkpoint_last.pt" \
    --tokenizer moses --moses-target-lang en ${BPE}
```

For interactive translation:
```bash
fairseq-interactive "data/bin_${VOCAB}" --task translation --user-dir examples/byte_level_bpe/gru_transformer \
    --path "checkpoints/${VOCAB}/checkpoint_last.pt" --input data/test.fr --tokenizer moses --moses-source-lang fr \
    --moses-target-lang en ${BPE} --buffer-size 1000 --max-tokens 10000
```

## Results
| Vocabulary | Model | BLEU |
|:----------:|:-----:|:----:|
| Joint BPE 16k | Transformer base 2+2 (w/ GRU) | 36.64 (36.72) |
| Joint BPE 4k | Transformer base 2+2 (w/ GRU) | 35.49 (36.10) |
| Joint BBPE 4k | Transformer base 2+2 (w/ GRU) | 35.61 (35.82) |
| Joint BPE 2k | Transformer base 2+2 (w/ GRU) | 34.87 (36.13) |
| Joint BBPE 2k | Transformer base 2+2 (w/ GRU) | 34.98 (35.43) |
| Characters | Transformer base 2+2 (w/ GRU) | 31.78 (33.30) |
| Bytes | Transformer base 2+2 (w/ GRU) | 31.57 (33.62) |