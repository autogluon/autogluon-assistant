# Condensed: Unsupervised Quality Estimation for Neural Machine Translation (Fomicheva et al., 2020)

Summary: This tutorial demonstrates implementing unsupervised quality estimation for neural machine translation based on Fomicheva et al. (2020). It covers two uncertainty quantification techniques: (1) a scoring method that applies dropout during inference to measure prediction confidence, and (2) a generation method that produces multiple translation hypotheses with dropout enabled to assess consistency. Key implementation details include enabling dropout at inference time with fairseq's `--retain-dropout` flag, specifying dropout modules, setting fixed seeds, and using Meteor for similarity computation. The code helps with preprocessing data, running stochastic forward passes, and aggregating uncertainty scores for translation quality estimation without reference translations.

*This is a condensed version that preserves essential implementation details and context.*

# Unsupervised Quality Estimation for Neural Machine Translation

This guide covers implementation details for reproducing results from [Unsupervised Quality Estimation for Neural Machine Translation (Fomicheva et al., 2020)](https://arxiv.org/abs/2005.10608).

## Setup Requirements

- mosesdecoder
- subword-nmt
- flores
- Models and test data from [MLQE dataset repository](https://github.com/facebookresearch/mlqe)

## Key Parameters

- `SRC_LANG`: source language
- `TGT_LANG`: target language
- `INPUT`: input file prefix
- `MODEL_DIR`: directory with NMT model and vocabularies
- `DROPOUT_N`: number of stochastic forward passes (30 in paper, but 10 is often sufficient)
- `GPU`: GPU ID for inference

## Implementation Process

### 1. Standard Translation

**Preprocess input:**
```bash
# Tokenize
perl $MOSES_DECODER/scripts/tokenizer/tokenizer.perl -threads 80 -a -l $LANG < $INPUT.$LANG > $TMP/preprocessed.tok.$LANG
# Apply BPE
python $BPE_ROOT/apply_bpe.py -c ${BPE} < $TMP/preprocessed.tok.$LANG > $TMP/preprocessed.tok.bpe.$LANG

# Binarize for faster translation
fairseq-preprocess --srcdict $MODEL_DIR/dict.$SRC_LANG.txt --tgtdict $MODEL_DIR/dict.$TGT_LANG.txt \
  --source-lang ${SRC_LANG} --target-lang ${TGT_LANG} --testpref $TMP/preprocessed.tok.bpe --destdir $TMP/bin --workers 4
```

**Translate:**
```bash
CUDA_VISIBLE_DEVICES=$GPU fairseq-generate $TMP/bin --path ${MODEL_DIR}/${SRC_LANG}-${TGT_LANG}.pt --beam 5 \
  --source-lang $SRC_LANG --target-lang $TGT_LANG --no-progress-bar --unkpen 5 > $TMP/fairseq.out
grep ^H $TMP/fairseq.out | cut -d- -f2- | sort -n | cut -f3- > $TMP/mt.out
```

**Post-process:**
```bash
sed -r 's/(@@ )| (@@ ?$)//g' < $TMP/mt.out | perl $MOSES_DECODER/scripts/tokenizer/detokenizer.perl \
  -l $TGT_LANG > $OUTPUT_DIR/mt.out
```

### 2. Uncertainty Estimation

#### Scoring Method

**Prepare repeated data:**
```bash
# Repeat source and MT output N times
python ${SCRIPTS}/scripts/uncertainty/repeat_lines.py -i $TMP/preprocessed.tok.bpe.$SRC_LANG -n $DROPOUT_N \
  -o $TMP/repeated.$SRC_LANG
python ${SCRIPTS}/scripts/uncertainty/repeat_lines.py -i $TMP/mt.out -n $DROPOUT_N -o $TMP/repeated.$TGT_LANG

# Binarize repeated data
fairseq-preprocess --srcdict ${MODEL_DIR}/dict.${SRC_LANG}.txt $TGT_DIC --source-lang ${SRC_LANG} \
  --target-lang ${TGT_LANG} --testpref ${TMP}/repeated --destdir ${TMP}/bin-repeated
```

**Score with dropout enabled:**
```bash
CUDA_VISIBLE_DEVICES=${GPU} fairseq-generate ${TMP}/bin-repeated --path ${MODEL_DIR}/${LP}.pt --beam 5 \
  --source-lang $SRC_LANG --target-lang $TGT_LANG --no-progress-bar --unkpen 5 --score-reference --retain-dropout \
  --retain-dropout-modules '["TransformerModel","TransformerEncoder","TransformerDecoder","TransformerEncoderLayer"]' \
  TransformerDecoderLayer --seed 46 > $TMP/dropout.scoring.out

grep ^H $TMP/dropout.scoring.out | cut -d- -f2- | sort -n | cut -f2 > $TMP/dropout.scores
```

**Compute mean scores:**
```bash
python $SCRIPTS/scripts/uncertainty/aggregate_scores.py -i $TMP/dropout.scores -o $OUTPUT_DIR/dropout.scores.mean \
  -n $DROPOUT_N
```

#### Generation Method

**Generate multiple hypotheses with dropout:**
```bash
CUDA_VISIBLE_DEVICES=${GPU} fairseq-generate ${TMP}/bin-repeated --path ${MODEL_DIR}/${LP}.pt \
  --beam 5 --source-lang $SRC_LANG --target-lang $TGT_LANG --no-progress-bar --retain-dropout \
  --unkpen 5 --retain-dropout-modules TransformerModel TransformerEncoder TransformerDecoder \
  TransformerEncoderLayer TransformerDecoderLayer --seed 46 > $TMP/dropout.generation.out

grep ^H $TMP/dropout.generation.out | cut -d- -f2- | sort -n | cut -f3- > $TMP/dropout.hypotheses_
```

**Post-process and compute similarity:**
```bash
# Detokenize
sed -r 's/(@@ )| (@@ ?$)//g' < $TMP/dropout.hypotheses_ | perl $MOSES_DECODER/scripts/tokenizer/detokenizer.perl \
  -l $TGT_LANG > $TMP/dropout.hypotheses

# Compute similarity with Meteor
python meteor.py -i $TMP/dropout.hypotheses -m <path_to_meteor_installation> -n $DROPOUT_N -o \
  $OUTPUT_DIR/dropout.gen.sim.meteor
```

## Key Implementation Notes

- The `--retain-dropout` flag is critical for enabling dropout at inference time
- `--retain-dropout-modules` specifies which modules should have dropout applied
- Setting a fixed seed (46) ensures reproducibility
- Meteor is used to compute similarity between multiple hypotheses