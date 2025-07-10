# Condensed: Transformer with Pointer-Generator Network

Summary: This tutorial explains implementing a Transformer with Pointer-Generator Network that enables copying words from input to output. It covers: 1) extending pointer-generator networks to Transformers by interpolating attention distribution with vocabulary distribution; 2) handling out-of-vocabulary words through position markers; 3) key implementation features including vocabulary creation with position markers, text preprocessing to replace unknown tokens, model training with specific parameters (alignment-heads, alignment-layer), and postprocessing to restore copied words. This approach is particularly valuable for tasks requiring copying input elements and when working with limited vocabularies.

*This is a condensed version that preserves essential implementation details and context.*

# Transformer with Pointer-Generator Network

This model incorporates a pointing mechanism in the Transformer that facilitates copying input words to the output, as described in [Enarvi et al. (2020)](https://www.aclweb.org/anthology/2020.nlpmc-1.4/).

## Implementation Details

- Extends the pointer-generator network concept from [See et al. (2017)](https://arxiv.org/abs/1704.04368) to Transformer models
- Interpolates attention distribution over input words with normal output distribution over vocabulary
- Enables generating words from input even if they're not in vocabulary (especially helpful with small vocabularies)
- Implementation differs from See et al. by handling OOV words through pre/post-processing rather than modifying the core architecture

## Usage Steps

### 1. Create vocabulary with source position markers

```bash
vocab_size=10000
position_markers=1000
export LC_ALL=C
cat train.src train.tgt |
  tr -s '[:space:]' '\n' |
  sort |
  uniq -c |
  sort -k1,1bnr -k2 |
  head -n "$((vocab_size - 4))" |
  awk '{ print $2 " " $1 }' >dict.pg.txt
python3 -c "[print('<unk-{}> 0'.format(n)) for n in range($position_markers)]" >>dict.pg.txt
```

### 2. Preprocess text data
Replace `<unk>` tokens with position-specific markers (`<unk-0>`, `<unk-1>`, etc.) using the provided `preprocess.py` script.

### 3. Train the model
Critical parameters:
- `--source-position-markers`: Number of special tokens
- `--alignment-heads` and `--alignment-layer`: Select attention distribution for pointing

### 4. Generate and postprocess text
- Preprocess input text the same way as training data
- Use `postprocess.py` to replace any `<unk-N>` tokens in output with the corresponding words from position N in the original input

This approach allows the model to copy out-of-vocabulary words from input to output while keeping implementation self-contained in the model file.