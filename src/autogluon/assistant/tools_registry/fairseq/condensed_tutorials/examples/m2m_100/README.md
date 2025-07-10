# Condensed: Beyond English-Centric Multilingual Machine Translation

Summary: This tutorial demonstrates implementing a Many-to-Many multilingual translation model supporting 100 languages using fairseq. It covers comprehensive data preprocessing techniques (deduplication, frequency cleaning, SentencePiece tokenization), model implementation details for different parameter sizes (418M, 1.2B, and 12B), and pipeline parallelism configurations for various GPU setups. The tutorial provides complete code for data preparation, model training, inference with different GPU configurations, and evaluation using BLEU scores. Key functionalities include direct translation between non-English language pairs, model parallelism for large models, and proper tokenization for multilingual contexts.

*This is a condensed version that preserves essential implementation details and context.*

# Beyond English-Centric Multilingual Machine Translation

## Introduction
This tutorial covers implementing a Many-to-Many multilingual translation model that can translate directly between any pair of 100 languages, with significant improvements (>10 BLEU) for non-English language pairs.

## Data Preparation

### Generation Data
Download evaluation datasets from various sources:
```bash
# WMT - using sacrebleu
sacrebleu -t wmt14 -l fr-en --echo src > wmt.test.fr-en.fr
sacrebleu -t wmt14 -l fr-en --echo ref > wmt.test.fr-en.en

# Other datasets: WAT, FLORES, TED, Autshumato, Tatoeba Challenge
# Download from their respective sources
```

### Training Data
Use a combination of [CCMatrix](https://arxiv.org/abs/1911.04944) and [CCAligned](https://arxiv.org/abs/1911.06154).

### Data Preprocessing
Critical preprocessing steps:
```bash
# Remove sentences with excessive punctuation
python /path/to/fairseq/examples/m2m_100/process_data/remove_too_much_punc.py 

# Deduplicate training data
paste /path/to/datadir/train.$src /path/to/datadir/train.$tgt | awk '!x[$0]++' > /path/to/datadir/train.dedup
cut -f1 /path/to/datadir/train.dedup > /path/to/datadir/train.$src
cut -f2 /path/to/datadir/train.dedup > /path/to/datadir/train.$tgt

# Remove evaluation data from training data
python /path/to/fairseq/examples/m2m_100/process_data/dedup_data.py 

# Frequency cleaning
wget https://dl.fbaipublicfiles.com/m2m_100/histograms.tar.gz 
tar -xvzf histograms.tar.gz
python /path/to/fairseq/examples/m2m_100/process_data/clean_histogram.py --src $src --tgt $tgt [...]

# Apply SentencePiece Model (SPM)
wget https://dl.fbaipublicfiles.com/m2m_100/spm.128k.model
python /path/to/fairseq/scripts/spm_encode.py --model spm.128k.model --output_format=piece [...]

# Length ratio cleaning
perl mosesdecoder/scripts/training/clean-corpus-n.perl --ratio 3 /path/to/training/data/train.spm.$src-$tgt $src $tgt /path/to/output/directory/train.spm.$src-$tgt 1 250

# Binarize data
wget https://dl.fbaipublicfiles.com/m2m_100/data_dict.128k.txt
fairseq-preprocess --source-lang $src --target-lang $tgt --testpref spm.$src.$tgt --thresholdsrc 0 --thresholdtgt 0 --destdir data_bin --srcdict data_dict.128k.txt --tgtdict data_dict.128k.txt
```

## Trained Models

### 418M and 1.2B Models
```bash
# Download model files
wget https://dl.fbaipublicfiles.com/m2m_100/model_dict.128k.txt
wget https://dl.fbaipublicfiles.com/m2m_100/language_pairs_small_models.txt 
wget https://dl.fbaipublicfiles.com/m2m_100/418M_last_checkpoint.pt  # 418M model
wget https://dl.fbaipublicfiles.com/m2m_100/1.2B_last_checkpoint.pt  # 1.2B model

# Generation command
fairseq-generate $binarized_data_path --batch-size 32 --path $path_to_model --fixed-dictionary model_dict.128k.txt -s en -t fr --remove-bpe 'sentencepiece' --beam 5 --task translation_multi_simple_epoch --lang-pairs language_pairs_small_models.txt --decoder-langtok --encoder-langtok src --gen-subset test > gen_out
```

### 12B Model
The 12B parameter model is available in different configurations based on GPU requirements:

**Model Download Links and Configuration**
| Configuration | 2 32GB GPUs | 4 16GB GPUs | 6 12GB GPUs | 8 8GB GPUs |
|:--|:--|:--|:--|:--|
| Last Checkpoint | [12b_last_chk_2_gpus.pt](https://dl.fbaipublicfiles.com/m2m_100/12b_last_chk_2_gpus.pt) | [12b_last_chk_4_gpus.pt](https://dl.fbaipublicfiles.com/m2m_100/12b_last_chk_4_gpus.pt) | [12b_last_chk_6_gpus.pt](https://dl.fbaipublicfiles.com/m2m_100/12b_last_chk_6_gpus.pt) | [12b_last_chk_8_gpus.pt](https://dl.fbaipublicfiles.com/m2m_100/12b_last_chk_8_gpus.pt) |
| Avg of last 5 checkpoints | [12b_avg5_chk_2_gpus.pt](https://dl.fbaipublicfiles.com/m2m_100/12b_avg5_chk_2_gpus.pt) | [12b_avg5_chk_4_gpus.pt](https://dl.fbaipublicfiles.com/m2m_100/12b_avg5_chk_4_gpus.pt) | [12b_avg5_chk_6_gpus.pt](https://dl.fbaipublicfiles.com/m2m_100/12b_avg5_chk_6_gpus.pt) | [12b_avg5_chk_8_gpus.pt](https://dl.fbaipublicfiles.com/m2m_100/12b_avg5_chk_8_gpus.pt) |
| Avg of last 10 checkpoints | [12b_avg10_chk_2_gpus.pt](https://dl.fbaipublicfiles.com/m2m_100/12b_avg10_chk_2_gpus.pt) | [12b_avg10_chk_4_gpus.pt](https://dl.fbaipublicfiles.com/m2m_100/12b_avg10_chk_4_gpus.pt) | [12b_avg10_chk_6_gpus.pt](https://dl.fbaipublicfiles.com/m2m_100/12b_avg10_chk_6_gpus.pt) | [12b_avg10_chk_8_gpus.pt](https://dl.fbaipublicfiles.com/m2m_100/12b_avg10_chk_8_gpus.pt) |

**Pipeline Arguments by GPU Configuration**
| Configuration | 2 32GB GPUs | 4 16GB GPUs | 6 12GB GPUs | 8 8GB GPUs |
|:--|:--|:--|:--|:--|
| `--pipeline-encoder-balance` | `[26]` | `[1,15,10]` | `[1,9,9,7]` | `[1,6,6,6,7]` |
| `--pipeline-encoder-devices` | `[0]` | `[0,1,0]` | `[0,1,2,0]` | `[0,4,5,1,0]` |
| `--pipeline-decoder-balance` | `[3,22,1]` | `[3,11,11,1]` | `[3,7,7,8,1]` | `[1,6,6,6,6,1]` |
| `--pipeline-decoder-devices` | `[0,1,0]` | `[0,2,3,0]` | `[0,3,4,5,0]` | `[0,2,6,7,3,0]` |

## Generation with M2M-100

### Encode using SentencePiece Model
```bash
fairseq=/path/to/fairseq
cd $fairseq
sacrebleu --echo src -l de-fr -t wmt19 | head -n 20 > raw_input.de-fr.de
sacrebleu --echo ref -l de-fr -t wmt19 | head -n 20 > raw_input.de-fr.fr
wget https://dl.fbaipublicfiles.com/m2m_100/spm.128k.model
for lang in de fr ; do
    python scripts/spm_encode.py \
        --model spm.128k.model \
        --output_format=piece \
        --inputs=raw_input.de-fr.${lang} \
        --outputs=spm.de-fr.${lang}
done
```

### Binarization
```bash
wget https://dl.fbaipublicfiles.com/m2m_100/data_dict.128k.txt
fairseq-preprocess \
    --source-lang de --target-lang fr \
    --testpref spm.de-fr \
    --thresholdsrc 0 --thresholdtgt 0 \
    --destdir data_bin \
    --srcdict data_dict.128k.txt --tgtdict data_dict.128k.txt
```

### Generation with 12B Model
```bash
wget https://dl.fbaipublicfiles.com/m2m_100/model_dict.128k.txt
wget https://dl.fbaipublicfiles.com/m2m_100/language_pairs.txt
wget https://dl.fbaipublicfiles.com/m2m_100/12b_last_chk_4_gpus.pt
fairseq-generate \
    data_bin \
    --batch-size 1 \
    --path 12b_last_chk_4_gpus.pt \
    --fixed-dictionary model_dict.128k.txt \
    -s de -t fr \
    --remove-bpe 'sentencepiece' \
    --beam 5 \
    --task translation_multi_simple_epoch \
    --lang-pairs language_pairs.txt \
    --decoder-langtok --encoder-langtok src \
    --gen-subset test \
    --fp16 \
    --dataset-impl mmap \
    --distributed-world-size 1 --distributed-no-spawn \
    --pipeline-model-parallel \
    --pipeline-chunks 1 \
    --pipeline-encoder-balance '[1,15,10]' \
    --pipeline-encoder-devices '[0,1,0]' \
    --pipeline-decoder-balance '[3,11,11,1]' \
    --pipeline-decoder-devices '[0,2,3,0]' > gen_out
```

## Evaluation
```bash
# Tokenization
cd ${fairseq}/examples/m2m_100
cat ${fairseq}/gen_out | grep -P "^H" | sort -V | cut -f 3- | sh tok.sh fr > hyp
cat ${fairseq}/raw_input.de-fr.fr | sh tok.sh fr > ref

# Calculate BLEU
sacrebleu -tok 'none' ref < hyp
```