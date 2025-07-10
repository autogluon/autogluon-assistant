# Condensed: LASER  Language-Agnostic SEntence Representations

Summary: This tutorial explains LASER, a library for multilingual sentence embeddings, focusing on implementation details for training. It covers data preparation using fairseq's preprocessing tools, JSON configuration file structure for multilingual datasets, and provides a complete training command with hyperparameters for LSTM-based models. The tutorial highlights key applications including cross-lingual document classification, parallel sentence mining (WikiMatrix), bitext mining, cross-lingual NLI, and multilingual similarity search. Developers can use this guide to implement multilingual sentence embedding models with specific architecture details (5-layer bidirectional encoder, 512 hidden size) and training parameters for cross-lingual NLP tasks.

*This is a condensed version that preserves essential implementation details and context.*

# LASER: Language-Agnostic SEntence Representations

LASER is a library for multilingual sentence embeddings. This guide covers training implementation details.

## Data Preparation and Configuration

1. Binarize your data using fairseq's preprocessing tools
2. Create a JSON config file with this structure:

```json
{
  "src_vocab": "/path/to/spm.src.cvocab",
  "tgt_vocab": "/path/to/spm.tgt.cvocab",
  "train": [
    {
      "type": "translation",
      "id": 0,
      "src": "/path/to/srclang1-tgtlang0/train.srclang1",
      "tgt": "/path/to/srclang1-tgtlang0/train.tgtlang0"
    },
    // Additional language pairs...
  ],
  "valid": [
    {
      "type": "translation",
      "id": 0,
      "src": "/unused",
      "tgt": "/unused"
    }
  ]
}
```

**Note**: `id` represents the target language ID and paths should point to binarized fairseq dataset files.

## Training Command

```bash
fairseq-train \
  /path/to/configfile.json \
  --user-dir examples/laser/laser_src \
  --log-interval 100 --log-format simple \
  --task laser --arch laser_lstm \
  --save-dir . \
  --optimizer adam \
  --lr 0.001 \
  --lr-scheduler inverse_sqrt \
  --clip-norm 5 \
  --warmup-updates 90000 \
  --update-freq 2 \
  --dropout 0.0 \
  --encoder-dropout-out 0.1 \
  --max-tokens 2000 \
  --max-epoch 50 \
  --encoder-bidirectional \
  --encoder-layers 5 \
  --encoder-hidden-size 512 \
  --decoder-layers 1 \
  --decoder-hidden-size 2048 \
  --encoder-embed-dim 320 \
  --decoder-embed-dim 320 \
  --decoder-lang-embed-dim 32 \
  --warmup-init-lr 0.001 \
  --disable-validation
```

## Key Applications

LASER embeddings can be used for various cross-lingual tasks without task-specific optimization:

- Cross-lingual document classification
- WikiMatrix: Mining 135M parallel sentences across 1620 language pairs
- Bitext mining using BUCC corpus
- Cross-lingual NLI using XNLI corpus
- Multilingual similarity search
- Sentence embedding for arbitrary text files

Implementation details for these applications are available in the `tasks` directory of the LASER repository.