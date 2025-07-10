# Condensed: Language Models not just for Pre-training: Fast Online Neural Noisy Channel Modeling

Summary: This tutorial demonstrates implementing fast noisy channel modeling for neural machine translation using Fairseq. It covers Bayesian-based scoring that combines direct translation models with channel models and language models. Key features include: (1) implementation of the noisy channel approach using P(y|x) = P(x|y) * P(y) / P(x), (2) speed optimization techniques like smaller channel models, reduced vocabulary, and fewer beam candidates, and (3) complete code examples for German-English and Romanian-English translation with model download links. The tutorial provides practical commands for model generation, preprocessing, and BLEU calculation, making it valuable for implementing efficient neural machine translation systems.

*This is a condensed version that preserves essential implementation details and context.*

# Fast Online Neural Noisy Channel Modeling

## Implementation Overview

This tutorial demonstrates how to implement fast noisy channel modeling for neural machine translation, based on [Yee et al. (2019)](https://www.aclweb.org/anthology/D19-1571.pdf) and [Bhosale et al. (2020)](http://www.statmt.org/wmt20/pdf/2020.wmt-1.68.pdf).

## Noisy Channel Modeling Core Concept

The approach applies Bayes Rule to predict `P(y|x)`:
```
P(y|x) = P(x|y) * P(y) / P(x)
```
Where:
- `P(x|y)`: channel model (predicts source given target)
- `P(y)`: language model over the target
- `P(x)`: constant for all `y` (not modeled)

During beam search, candidates are scored with:
```
(1 / t) * log(P(y|x) + (1 / s) * (λ1 * log(P(x|y)) + λ2 * log(P(y)))
```
Where:
- `t`: Target prefix length
- `s`: Source length
- `λ1`: Channel model weight
- `λ2`: Language model weight

## Implementation Examples

### Generation with Language Model

```sh
binarized_data=data_dir/binarized
direct_model=de_en_seed4.pt
lm_model=en_lm.pt
lm_data=lm_data
wget https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/direct_models/seed4.pt -O ${direct_model}
wget https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/lm_model/transformer_lm.pt -O ${lm_model}
mkdir -p ${lm_data}
wget https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/lm_model/lm_dict/dict.txt -O ${lm_data}/dict.txt

fairseq-generate ${binarized_data} \
    --user-dir examples/fast_noisy_channel \
    --beam 5 \
    --path ${direct_model} \
    --lm-model ${lm_model} \
    --lm-data ${lm_data}  \
    --k2 10 \
    --combine-method lm_only \
    --task noisy_channel_translation \
    --lenpen 0.16 \
    --lm-wt 0.14 \
    --gen-subset valid \
    --remove-bpe \
    --fp16 \
    --batch-size 10
```

### Full Noisy Channel Generation

```sh
binarized_data=data_dir/binarized
direct_model=de_en_seed4.pt
lm_model=en_lm.pt
lm_data=lm_data
ch_model=en_de.big.seed4.pt
wget https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/direct_models/seed4.pt -O ${direct_model}
wget https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/lm_model/transformer_lm.pt -O ${lm_model}
mkdir -p ${lm_data}
wget https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/lm_model/lm_dict/dict.txt -O ${lm_data}/dict.txt
wget https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/channel_models/big.seed4.pt -O ${ch_model}

fairseq-generate ${binarized_data} \
    --user-dir examples/fast_noisy_channel \
    --beam 5 \
    --path ${direct_model} \
    --lm-model ${lm_model} \
    --lm-data ${lm_data}  \
    --channel-model ${ch_model} \
    --k2 10 \
    --combine-method noisy_channel \
    --task noisy_channel_translation \
    --lenpen 0.21 \
    --lm-wt 0.50 \
    --ch-wt 0.30 \
    --gen-subset test \
    --remove-bpe \
    --fp16 \
    --batch-size 1
```

## Speed Optimizations

[Bhosale et al. (2020)](http://www.statmt.org/wmt20/pdf/2020.wmt-1.68.pdf) introduces three key optimizations:

1. **Smaller channel models**: Use `Transformer Base` with 1 encoder and decoder layer each instead of `Transformer Big`

2. **Reduced output vocabulary**: Limit channel model vocabulary to source tokens plus most frequent tokens
   ```
   --channel-scoring-type src_vocab --top-k-vocab 500
   ```

3. **Fewer candidates per beam**: Reduce the `--k2` parameter value

These optimizations significantly improve speed with minimal accuracy loss.

# Fast Noisy Channel Generation for Machine Translation

## Fast Noisy Channel Generation for German-English Translation

This section provides implementation details for fast noisy channel generation using fairseq with direct, channel, and language models.

### Setup and Model Downloads

```sh
binarized_data=data_dir/binarized
direct_model=de_en_seed4.pt
lm_model=en_lm.pt
lm_data=lm_data
small_ch_model=en_de.base_1_1.seed4.pt

# Download models
wget https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/direct_models/seed4.pt -O ${direct_model}
wget https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/lm_model/transformer_lm.pt -O ${lm_model}
mkdir -p ${lm_data}
wget https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/lm_model/lm_dict/dict.txt -O ${lm_data}/dict.txt
wget https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/channel_models/base_1_1.seed4.pt -O ${small_ch_model}
```

### Generation Command

```sh
k2=3
lenpen=0.23
lm_wt=0.58
bw_wt=0.26

fairseq-generate ${binarized_data} \
    --user-dir examples/fast_noisy_channel \
    --beam 5 \
    --path ${direct_model} \
    --lm-model ${lm_model} \
    --lm-data ${lm_data}  \
    --channel-model ${small_ch_model} \
    --k2 ${k2} \
    --combine-method noisy_channel \
    --task noisy_channel_translation \
    --lenpen ${lenpen} \
    --lm-wt ${lm_wt} \
    --ch-wt ${bw_wt} \
    --gen-subset test \
    --remove-bpe \
    --fp16 \
    --batch-size 50 \
    --channel-scoring-type src_vocab --top-k-vocab 500
```

## Test Data Preprocessing

Script for preprocessing and binarizing test sets:

```sh
FAIRSEQ=/path/to/fairseq
cd $FAIRSEQ
SCRIPTS=$FAIRSEQ/mosesdecoder/scripts
# Clone Moses if needed
if [ ! -d "${SCRIPTS}" ]; then
    git clone https://github.com/moses-smt/mosesdecoder.git
fi
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
NORMALIZE=$SCRIPTS/tokenizer/normalize-punctuation.perl

s=de
t=en
test=wmt18

mkdir -p data_dir

# Tokenization (with special handling for Romanian)
if [ $s == "ro" ] ; then
    sacrebleu -t $test -l $s-$t --echo src | \
        $NORMALIZE -l $s | \
        python normalise-romanian.py | \
        python remove-diacritics.py | \
        $TOKENIZER -l $s -a -q > data_dir/$test.$s-$t.$s
else
    sacrebleu -t $test -l $s-$t --echo src | perl $NORMALIZE -l $s | perl $TOKENIZER -threads 8 -a -l $s > data_dir/$test.$s-$t.$s
fi

sacrebleu -t $test -l $s-$t --echo ref | perl $NORMALIZE -l $t | perl $TOKENIZER -threads 8 -a -l $t > data_dir/$test.$s-$t.$t

# Apply BPE and binarize
src_bpe_code=/path/to/source/language/bpe/code
tgt_bpe_code=/path/to/target/language/bpe/code
src_dict=/path/to/source/language/dict
tgt_dict=/path/to/target/language/dict

# Setup FastBPE
FASTBPE=$FAIRSEQ/fastBPE
if [ ! -d "${FASTBPE}" ] ; then
    git clone https://github.com/glample/fastBPE.git
    g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast
fi

${FASTBPE}/fast applybpe data_dir/bpe.$test.$s-$t.$s data_dir/$test.$s-$t.$s ${src_bpe_code}
${FASTBPE}/fast applybpe data_dir/bpe.$test.$s-$t.$t data_dir/$test.$s-$t.$t ${tgt_bpe_code}

fairseq-preprocess -s $s -t $t \
    --testpref data_dir/bpe.$test.$s-$t \
    --destdir data_dir/binarized \
    --srcdict ${src_dict} \
    --tgtdict ${tgt_dict}
```

## Calculating BLEU

```sh
DETOKENIZER=$SCRIPTS/tokenizer/detokenizer.perl
cat ${generation_output} | grep -P "^H" | sort -V | cut -f 3- | $DETOKENIZER -l $t -q -a | sacrebleu -t $test -l $s-$t
```

## Romanian-English Translation

### BPE and Dictionary
- Joint BPE vocabulary of 18K types
- BPE Code: [joint_bpe_18k](https://dl.fbaipublicfiles.com/fast_noisy_channel/ro_en/bpe_18k)
- Dictionary: [dict](https://dl.fbaipublicfiles.com/fast_noisy_channel/ro_en/dict)

### Direct Models (Transformer-Big)
| Seed | Model |
|----|----|
| 2 | [ro_en_seed2.pt](https://dl.fbaipublicfiles.com/fast_noisy_channel/ro_en/direct_models/seed2.pt) |
| 4 | [ro_en_seed4.pt](https://dl.fbaipublicfiles.com/fast_noisy_channel/ro_en/direct_models/seed4.pt) |
| 6 | [ro_en_seed6.pt](https://dl.fbaipublicfiles.com/fast_noisy_channel/ro_en/direct_models/seed6.pt) |

### Channel Models
Best hyperparameters from validation set (wmt16/dev) using beam 5:

| Model Size | Lenpen | LM Weight | CH Weight | Seed 2 | Seed 4 | Seed 6 |
|----|----|----|----|----|----|----|
| `big` | 0.84 | 0.64 | 0.56 | [big.seed2.pt](https://dl.fbaipublicfiles.com/fast_noisy_channel/ro_en/channel_models/big.seed2.pt) | [big.seed2.pt](https://dl.fbaipublicfiles.com/fast_noisy_channel/ro_en/channel_models/big.seed2.pt) | [big.seed2.pt](https://dl.fbaipublicfiles.com/fast_noisy_channel/ro_en/channel_models/big.seed2.pt) |
| `base_1_1` | 0.63 | 0.40 | 0.37 | [base_1_1.seed2.pt](https://dl.fbaipublicfiles.com/fast_noisy_channel/ro_en/channel_models/base_1_1.seed2.pt) | [base_1_1.seed4.pt](https://dl.fbaipublicfiles.com/fast_noisy_channel/ro_en/channel_models/base_1_1.seed4.pt) | [base_1_1.seed6.pt](https://dl.fbaipublicfiles.com/fast_noisy_channel/ro_en/channel_models/base_1_1.seed6.pt) |

### Language Model
- Trained on de-duplicated English Newscrawl data (2007-2018): 186M sentences, 4.5B words
- Model: [transformer_en_lm](https://dl.fbaipublicfiles.com/fast_noisy_channel/ro_en/lm_model/transformer_lm.pt)
- Dictionary: [lm_data](https://dl.fbaipublicfiles.com/fast_noisy_channel/ro_en/lm_model/lm_dict)

## German-English Translation

### BPE and Dictionaries
| Resource | Path |
|----------|------|
| Source BPE Code | [de_bpe_code_24K](https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/de_bpe_code_24K) |
| Target BPE Code | [en_bpe_code_24K](https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/en_bpe_code_24K) |
| Source Dictionary | [de_dict](https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/de_dict) |
| Target Dictionary | [en_dict](https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/en_dict) |

### Direct Models (Transformer-Big)
Trained on WMT'19 data (26.8M sentence pairs) after filtering:

| Seed | Model |
|:----:|----|
| 4 | [de_en_seed4.pt](https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/direct_models/seed4.pt) |
| 5 | [de_en_seed5.pt](https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/direct_models/seed5.pt) |
| 6 | [de_en_seed6.pt](https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/direct_models/seed6.pt) |

# Channel Models and Language Model Implementation


...(truncated)