# Condensed: Jointly Learning to Align and Translate with Transformer Models (Garg et al., 2019)

Summary: This tutorial demonstrates implementing the "Jointly Learning to Align and Translate with Transformer Models" paper using Fairseq. It covers: (1) training a neural machine translation system with explicit alignment learning, (2) preprocessing WMT'18 En-De data with FastAlign to generate alignments, and (3) training a transformer model that incorporates alignment information. Key functionalities include data preparation, alignment generation, model training with alignment-aware architecture, and inference with alignment output. The tutorial provides complete command-line instructions for each step, including hyperparameter settings for both standard and large-batch training scenarios.

*This is a condensed version that preserves essential implementation details and context.*

# Jointly Learning to Align and Translate with Transformer Models

Implementation of the paper [Jointly Learning to Align and Translate with Transformer Models (Garg et al., 2019)](https://arxiv.org/abs/1909.02074).

## Training Process for WMT'18 En-De

### 1. Data Preparation
```bash
./prepare-wmt18en2de_no_norm_no_escape_no_agressive.sh
```

### 2. Generate Alignments with FastAlign
```bash
git clone git@github.com:clab/fast_align.git
pushd fast_align
mkdir build && cd build && cmake .. && make
popd
ALIGN=fast_align/build/fast_align
paste bpe.32k/train.en bpe.32k/train.de | awk -F '\t' '{print $1 " ||| " $2}' > bpe.32k/train.en-de
$ALIGN -i bpe.32k/train.en-de -d -o -v > bpe.32k/train.align
```

### 3. Preprocess with Alignments
```bash
fairseq-preprocess \
    --source-lang en --target-lang de \
    --trainpref bpe.32k/train \
    --validpref bpe.32k/valid \
    --testpref bpe.32k/test \
    --align-suffix align \
    --destdir binarized/ \
    --joined-dictionary \
    --workers 32
```

### 4. Train the Model
```bash
fairseq-train \
    binarized \
    --arch transformer_wmt_en_de_big_align --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --activation-fn relu\
    --lr 0.0002 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
    --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.0 \
    --max-tokens 3500 --label-smoothing 0.1 \
    --save-dir ./checkpoints --log-interval 1000 --max-update 60000 \
    --keep-interval-updates -1 --save-interval-updates 0 \
    --load-alignments --criterion label_smoothed_cross_entropy_with_alignment \
    --fp16
```

**Important notes:**
- `--fp16` requires CUDA 9.1+ and a Volta GPU or newer
- For large batch training (8 GPUs):
  - Add `--update-freq 8` to simulate 64 GPUs
  - Increase learning rate to ~0.0007

### 5. Generate Alignments (BPE level)
```bash
fairseq-generate \
    binarized --gen-subset test --print-alignment \
    --source-lang en --target-lang de \
    --path checkpoints/checkpoint_best.pt --beam 5 --nbest 1
```

### 6. Additional Resources
For alignment test set preparation, BPE-to-token alignment conversion, alignment symmetrization, and AER metric evaluation, see: [https://github.com/lilt/alignment-scripts](https://github.com/lilt/alignment-scripts)