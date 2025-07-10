# Condensed: Pretraining RoBERTa using your own data

Summary: This tutorial demonstrates how to pretrain RoBERTa on custom data using fairseq. It covers three key implementation steps: (1) preprocessing text data with GPT-2 BPE encoding and binarization, (2) training RoBERTa with configurable batch sizes and GPU setups, including guidance on batch size/learning rate relationships and gradient accumulation for different hardware configurations, and (3) loading the pretrained model in Python. The tutorial helps with tasks like custom language model pretraining, working with WikiText datasets, and optimizing training parameters for available computational resources.

*This is a condensed version that preserves essential implementation details and context.*

# Pretraining RoBERTa Using Your Own Data

## 1) Preprocess the Data

Data should follow the language modeling format with documents separated by empty lines. Lines are concatenated as a 1D text stream during training.

### Example with WikiText-103 dataset:

```bash
# Download dataset
wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip
unzip wikitext-103-raw-v1.zip

# Encode with GPT-2 BPE
mkdir -p gpt2_bpe
wget -O gpt2_bpe/encoder.json https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json
wget -O gpt2_bpe/vocab.bpe https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe
for SPLIT in train valid test; do \
    python -m examples.roberta.multiprocessing_bpe_encoder \
        --encoder-json gpt2_bpe/encoder.json \
        --vocab-bpe gpt2_bpe/vocab.bpe \
        --inputs wikitext-103-raw/wiki.${SPLIT}.raw \
        --outputs wikitext-103-raw/wiki.${SPLIT}.bpe \
        --keep-empty \
        --workers 60; \
done

# Preprocess/binarize data
wget -O gpt2_bpe/dict.txt https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt
fairseq-preprocess \
    --only-source \
    --srcdict gpt2_bpe/dict.txt \
    --trainpref wikitext-103-raw/wiki.train.bpe \
    --validpref wikitext-103-raw/wiki.valid.bpe \
    --testpref wikitext-103-raw/wiki.test.bpe \
    --destdir data-bin/wikitext-103 \
    --workers 60
```

## 2) Train RoBERTa Base

```bash
DATA_DIR=data-bin/wikitext-103

fairseq-hydra-train -m --config-dir examples/roberta/config/pretraining \
--config-name base task.data=$DATA_DIR
```

**Important Notes:**
- You can resume training from the released RoBERTa base model by adding `checkpoint.restore_file=/path/to/roberta.base/model.pt`
- Default configuration assumes 8x32GB V100 GPUs with:
  - `dataset.batch_size=16` sequences per GPU
  - `optimization.update_freq=16` for gradient accumulation
  - Total effective batch size: 2048 sequences
- For fewer/smaller GPUs: reduce `dataset.batch_size` and increase `dataset.update_freq`
- For more GPUs: decrease `dataset.update_freq` to increase training speed

**Batch Size and Learning Rate Relationship:**

| Batch Size | Peak Learning Rate |
|------------|-------------------|
| 256        | 0.0001            |
| 2048       | 0.0005            |
| 8192       | 0.0007            |

## 3) Load Your Pretrained Model

```python
from fairseq.models.roberta import RobertaModel
roberta = RobertaModel.from_pretrained('checkpoints', 'checkpoint_best.pt', 'path/to/data')
assert isinstance(roberta.model, torch.nn.Module)
```