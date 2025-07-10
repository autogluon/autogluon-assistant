# Condensed: Masked Language Modeling and the Distributional Hypothesis: Order Word Matters Pre-training for Little

Summary: This tutorial provides implementation guidance for working with RoBERTa models pre-trained on word-shuffled corpora. It covers loading and using various pre-trained models (standard RoBERTa, n-gram shuffled variants, and positional embedding-free models) using the Fairseq library. The tutorial helps with tasks requiring masked language modeling and word order experimentation, including model loading, evaluation, and fine-tuning. Key features include code snippets for model initialization, special handling instructions for no-positional-embeddings models, performance benchmarks across GLUE tasks, and links to pre-trained model weights for different word-order configurations.

*This is a condensed version that preserves essential implementation details and context.*

# Masked Language Modeling and Word Order Pre-training

## Pre-trained Models

Various RoBERTa base models trained on word-shuffled variants of BookWiki corpus (16GB):

| Model Type | Description | Download Link |
|------------|-------------|--------------|
| `roberta.base.orig` | Standard RoBERTa on natural corpus | [Download](https://dl.fbaipublicfiles.com/unnatural_pretraining/roberta.base.orig.tar.gz) |
| `roberta.base.shuffle.n1` | Trained on n=1 gram shuffled data | [Download](https://dl.fbaipublicfiles.com/unnatural_pretraining/roberta.base.shuffle.n1.tar.gz) |
| `roberta.base.shuffle.n2-n4` | Trained on n=2-4 gram shuffled data | [Download links in original doc] |
| `roberta.base.nopos` | Without positional embeddings | [Download](https://dl.fbaipublicfiles.com/unnatural_pretraining/roberta.base.nopos.tar.gz) |

## Implementation Usage

```python
# Download and extract model
wget https://dl.fbaipublicfiles.com/unnatural_pretraining/roberta.base.shuffle.n1.tar.gz
tar -xzvf roberta.base.shuffle.n1.tar.gz

# Get dictionary files
cd roberta.base.shuffle.n1.tar.gz
wget -O dict.txt https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt
wget -O encoder.json https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json
wget -O vocab.bpe https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe
cd ..

# Load model in fairseq
from fairseq.models.roberta import RobertaModel
roberta = RobertaModel.from_pretrained('/path/to/roberta.base.shuffle.n1', checkpoint_file='model.pt')
roberta.eval()  # disable dropout (or leave in train mode to finetune)
```

## Important Notes

- **No-positional-embeddings model**: The `roberta.base.nopos` model requires special handling. Set `use_positional_embeddings=False` or `no_token_positional_embeddings=True` when constructing the model before loading weights.

- Models were trained using Fairseq commit: `62cff008ebeeed855093837507d5e6bf52065ee6`

- A [Google Colab notebook](https://colab.research.google.com/drive/1IJDVfNVWdvRfLjphQKBGzmob84t-OXpm) is available for demonstration.

## Results

GLUE and PAWS dev set performance (median of 5 seeds, single-task fine-tuning):

| Model | CoLA | MNLI | MRPC | PAWS | QNLI | QQP | RTE | SST-2 |
|-------|------|------|------|------|------|-----|-----|-------|
| `roberta.base.orig` | 61.4 | 86.11 | 89.19 | 94.46 | 92.53 | 91.26 | 74.64 | 93.92 |
| `roberta.base.shuffle.n1` | 35.15 | 82.64 | 86 | 89.97 | 89.02 | 91.01 | 69.02 | 90.47 |
| `roberta.base.shuffle.n2` | 54.37 | 83.43 | 86.24 | 93.46 | 90.44 | 91.36 | 70.83 | 91.79 |

Fine-tuned MNLI models are also available for evaluation with accuracy scores provided in the original documentation.