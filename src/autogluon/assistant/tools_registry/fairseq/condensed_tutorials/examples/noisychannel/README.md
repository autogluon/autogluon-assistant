# Condensed: Simple and Effective Noisy Channel Modeling for Neural Machine Translation (Yee et al., 2019)

Summary: This tutorial demonstrates implementing noisy channel modeling for neural machine translation using fairseq. It covers how to set up pre-trained German-English translation models (forward, channel, and language models), and provides code for three reranking approaches: combining P(T|S), P(S|T), and P(T); using only P(T|S) and P(T); or applying pre-configured hyperparameters. The implementation includes hyperparameter tuning with configurable beam size, batch processing, and model weighting. Developers can use this to build translation systems with improved accuracy through noisy channel modeling, particularly for German-English translation tasks.

*This is a condensed version that preserves essential implementation details and context.*

# Noisy Channel Modeling for Neural Machine Translation

## Pre-trained Models

| Model | Description | Download |
|---|---|---|
| `transformer.noisychannel.de-en` | De->En Forward Model | [download](https://dl.fbaipublicfiles.com/fairseq/models/noisychannel/forward_de2en.tar.bz2) |
| `transformer.noisychannel.en-de` | En->De Channel Model | [download](https://dl.fbaipublicfiles.com/fairseq/models/noisychannel/backward_en2de.tar.bz2) |
| `transformer_lm.noisychannel.en` | En Language model | [download](https://dl.fbaipublicfiles.com/fairseq/models/noisychannel/reranking_en_lm.tar.bz2) |

Test Data: [newstest_wmt17](https://dl.fbaipublicfiles.com/fairseq/models/noisychannel/wmt17test.tar.bz2)

## Implementation

### Setup
```bash
mkdir rerank_example
curl https://dl.fbaipublicfiles.com/fairseq/models/noisychannel/forward_de2en.tar.bz2 | tar xvjf - -C rerank_example
curl https://dl.fbaipublicfiles.com/fairseq/models/noisychannel/backward_en2de.tar.bz2 | tar xvjf - -C rerank_example
curl https://dl.fbaipublicfiles.com/fairseq/models/noisychannel/reranking_en_lm.tar.bz2 | tar xvjf - -C rerank_example
curl https://dl.fbaipublicfiles.com/fairseq/models/noisychannel/wmt17test.tar.bz2 | tar xvjf - -C rerank_example
```

### Key Configuration Parameters
```bash
beam=50                # Beam size for generation
num_trials=1000        # Number of trials for hyperparameter tuning
data_dir=rerank_example/hyphen-splitting-mixed-case-wmt17test-wmt14bpe
lm=rerank_example/lm/checkpoint_best.pt
lm_bpe_code=rerank_example/lm/bpe32k.code
lm_dict=rerank_example/lm/dict.txt
batch_size=32
bw=rerank_example/backward_en2de.pt    # Channel model
fw=rerank_example/forward_de2en.pt     # Forward model
```

### Reranking Methods

#### 1. Reranking with P(T|S), P(S|T) and P(T)
```bash
python examples/noisychannel/rerank_tune.py $data_dir \
    --tune-param lenpen weight1 weight3 \
    --lower-bound 0 0 0 --upper-bound 3 3 3 --data-dir-name $data_dir_name \
    --num-trials $num_trials --source-lang de --target-lang en --gen-model $fw \
    -n $beam --batch-size $batch_size --score-model2 $fw --score-model1 $bw \
    --backwards1 --weight2 1 \
    -lm $lm --lm-dict $lm_dict --lm-name en_newscrawl --lm-bpe-code $lm_bpe_code
```

#### 2. Reranking with P(T|S) and P(T) only
```bash
python examples/noisychannel/rerank_tune.py $data_dir \
    --tune-param lenpen weight3 \
    --lower-bound 0 0 --upper-bound 3 3 --data-dir-name $data_dir_name \
    --num-trials $num_trials --source-lang de --target-lang en --gen-model $fw \
    -n $beam --batch-size $batch_size --score-model1 $fw \
    -lm $lm --lm-dict $lm_dict --lm-name en_newscrawl --lm-bpe-code $lm_bpe_code
```

#### 3. Using pre-configured hyperparameters
```bash
python examples/noisychannel/rerank.py $data_dir \
    --lenpen 0.269 --weight1 1 --weight2 0.929 --weight3 0.831 \
    --data-dir-name $data_dir_name --source-lang de --target-lang en --gen-model $fw \
    -n $beam --batch-size $batch_size --score-model2 $fw --score-model1 $bw --backwards1 \
    -lm $lm --lm-dict $lm_dict --lm-name en_newscrawl --lm-bpe-code $lm_bpe_code
```

## Citation
```bibtex
@inproceedings{yee2019simple,
  title = {Simple and Effective Noisy Channel Modeling for Neural Machine Translation},
  author = {Kyra Yee and Yann Dauphin and Michael Auli},
  booktitle = {Conference on Empirical Methods in Natural Language Processing},
  year = {2019},
}
```