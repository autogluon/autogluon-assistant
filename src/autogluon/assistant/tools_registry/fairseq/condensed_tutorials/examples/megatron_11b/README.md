# Condensed: Megatron-11b

Summary: This tutorial covers implementing and using Megatron-11b, an 11B parameter language model based on Megatron-LM. It details model parallel training across 8 GPUs using Fairseq, including specific architecture parameters (3072 embed_dim, 72 layers, 32 attention heads) and training hyperparameters. The tutorial provides complete code for training with model parallelism, downloading and evaluating the pre-trained model on Wikitext-103, data preprocessing (detokenization, BPE encoding, binarization), and perplexity calculation with renormalization. Key functionalities include distributed training, memory-efficient FP16 operations, and model parallel inference.

*This is a condensed version that preserves essential implementation details and context.*

# Megatron-11b

Megatron-11b is an 11B parameter unidirectional language model based on [Megatron-LM](https://arxiv.org/pdf/1909.08053.pdf), trained using intra-layer model parallelism across 8 GPUs. It uses the same data and byte-pair encoding as RoBERTa.

## Model Specifications

- **Parameters**: 11B
- **Architecture**:
  - `embed_dim`: 3072
  - `ffn_dim`: 18432 (3072 * 6)
  - `layers`: 72
  - `attention heads`: 32
- **Training details**:
  - Batch size: 512
  - Updates: 300,000
  - Peak learning rate: 1.5e-04
  - LR scheduler: inverse_sqrt
  - Clip norm: 0.0

## Model Parallel Training Command

```bash
fairseq-train <DATA_PATH> \
  --distributed-world-size 8  \
  --memory-efficient-fp16 \
  --num-workers 2 \
  --model-parallel-size 8 \
  --criterion vocab_parallel_cross_entropy \
  --task language_modeling \
  --sample-break-mode none \
  --tokens-per-sample 1024 \
  --arch transformer_lm_megatron_11b \
  --share-decoder-input-output-embed \
  --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-08 --clip-norm 0.0 \
  --lr-scheduler inverse_sqrt --lr 0.00015 \
  --warmup-updates 3000 --weight-decay 0.01 \
  --dropout 0.1 --attention-dropout 0.1 \
  --batch-size 2 \
  --max-update 300000;
```

**Note**: Tested on `DGX-1` with `8xV100-32Gb` GPUs.

## Evaluation on Wikitext-103

1. **Download model** (19GB):
   ```bash
   wget https://dl.fbaipublicfiles.com/fairseq/models/model_parallel/megatron_11b.tar.gz
   tar -xzvf megatron_11b.tar.gz
   ```

2. **Download Wikitext-103**:
   ```bash
   wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip
   unzip wikitext-103-raw-v1.zip
   ```

3. **Detokenize test data** (model expects raw input):
   ```bash
   python -m examples.megatron_11b.detok wikitext-103-raw/wiki.test.raw > wikitext-103-raw/wiki.test.detok
   ```

4. **Apply BPE encoding**:
   ```bash
   wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
   wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
   
   python -m examples.roberta.multiprocessing_bpe_encoder \
       --encoder-json encoder.json \
       --vocab-bpe vocab.bpe \
       --inputs "wikitext-103-raw/wiki.test.detok" \
       --outputs "wikitext-103-raw/wiki.test.bpe" \
       --workers 60;
   ```

5. **Binarize data**:
   ```bash
   fairseq-preprocess \
       --only-source \
       --testpref wikitext-103-raw/wiki.test.bpe \
       --srcdict megatron_11b/dict.txt \
       --destdir wikitext103-bin;
   ```

6. **Evaluate perplexity**:
   ```bash
   DATA_PATH=wikitext103-bin/
   fairseq-eval-lm \
     $DATA_PATH \
     --path megatron_11b/model.pt \
     --task language_modeling \
     --gen-subset test \
     --batch-size 8 \
     --criterion cross_entropy \
     --context-window 992 \
     --distributed-world-size 8 \
     --model-parallel-size 8;
   # Expected unnormalized PPL: 8.46
   ```

   **Important**: Perplexity must be renormalized due to detokenization and BPE:
   - Formula: `2 ^ (log_2(unnormalized_PPL) * (new_token_cnt / orig_token_cnt))`
   - For Wikitext-103: `2 ^ (log_2(8.46) * (270847 / 245566)) = 10.54`

## Results

**Wikitext-103 Performance**:
- Valid perplexity: 10.64
- Test perplexity: 10.54