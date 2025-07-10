# Condensed: Understanding Back-Translation at Scale (Edunov et al., 2018)

Summary: This tutorial demonstrates implementing back-translation for neural machine translation using fairseq. It covers loading pre-trained WMT'18 English-German translation models via torch.hub, and provides a complete workflow for training enhanced translation systems: building baseline models, creating reverse models for back-translation, generating synthetic parallel data from monolingual sources using sampling techniques, and combining original and back-translated data to train improved models. Key functionalities include model training, checkpoint averaging, data preprocessing, and evaluation with BLEU scores—essential techniques for improving machine translation quality through synthetic data augmentation.

*This is a condensed version that preserves essential implementation details and context.*

# Understanding Back-Translation at Scale (Edunov et al., 2018)

## Pre-trained Models

| Model | Description | Download |
|---|---|---|
| `transformer.wmt18.en-de` | Transformer (WMT'18 winner) | [download (.tar.gz)](https://dl.fbaipublicfiles.com/fairseq/models/wmt18.en-de.ensemble.tar.gz) |

## Usage with torch.hub

```bash
pip install subword_nmt sacremoses
```

```python
import torch

# Load the WMT'18 En-De ensemble
en2de_ensemble = torch.hub.load(
    'pytorch/fairseq', 'transformer.wmt18.en-de',
    checkpoint_file='wmt18.model1.pt:wmt18.model2.pt:wmt18.model3.pt:wmt18.model4.pt:wmt18.model5.pt',
    tokenizer='moses', bpe='subword_nmt')

# Translate
en2de_ensemble.translate('Hello world!')  # 'Hallo Welt!'
```

## Training Your Own Model

### 1. Prepare Data and Train Baseline Model

```bash
# Download and preprocess data
cd examples/backtranslation/
bash prepare-wmt18en2de.sh
cd ../..

# Binarize data
TEXT=examples/backtranslation/wmt18_en_de
fairseq-preprocess \
    --joined-dictionary \
    --source-lang en --target-lang de \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/wmt18_en_de --thresholdtgt 0 --thresholdsrc 0 \
    --workers 20

# Copy BPE code
cp examples/backtranslation/wmt18_en_de/code data-bin/wmt18_en_de/code
```

Train baseline model:
```bash
CHECKPOINT_DIR=checkpoints_en_de_parallel
fairseq-train --fp16 \
    data-bin/wmt18_en_de \
    --source-lang en --target-lang de \
    --arch transformer_wmt_en_de_big --share-all-embeddings \
    --dropout 0.3 --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 0.001 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --max-tokens 3584 --update-freq 16 \
    --max-update 30000 \
    --save-dir $CHECKPOINT_DIR
# Note: Adjust --update-freq based on GPU count
```

Average checkpoints and evaluate:
```bash
python scripts/average_checkpoints.py \
    --inputs $CHECKPOINT_DIR \
    --num-epoch-checkpoints 10 \
    --output $CHECKPOINT_DIR/checkpoint.avg10.pt

# Evaluate with sacrebleu
bash examples/backtranslation/sacrebleu.sh \
    wmt17 \
    en-de \
    data-bin/wmt18_en_de \
    data-bin/wmt18_en_de/code \
    $CHECKPOINT_DIR/checkpoint.avg10.pt
```

### 2. Back-translate Monolingual German Data

Train reverse model (German-English):
```bash
CHECKPOINT_DIR=checkpoints_de_en_parallel
fairseq-train --fp16 \
    data-bin/wmt18_en_de \
    --source-lang de --target-lang en \
    --arch transformer_wmt_en_de_big --share-all-embeddings \
    --dropout 0.3 --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 0.001 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --max-tokens 3584 --update-freq 16 \
    --max-update 30000 \
    --save-dir $CHECKPOINT_DIR
```

Prepare and process monolingual data:
```bash
cd examples/backtranslation/
bash prepare-de-monolingual.sh
cd ../..

# Binarize each shard
TEXT=examples/backtranslation/wmt18_de_mono
for SHARD in $(seq -f "%02g" 0 24); do \
    fairseq-preprocess \
        --only-source \
        --source-lang de --target-lang en \
        --joined-dictionary \
        --srcdict data-bin/wmt18_en_de/dict.de.txt \
        --testpref $TEXT/bpe.monolingual.dedup.${SHARD} \
        --destdir data-bin/wmt18_de_mono/shard${SHARD} \
        --workers 20; \
    cp data-bin/wmt18_en_de/dict.en.txt data-bin/wmt18_de_mono/shard${SHARD}/; \
done
```

Generate back-translations using sampling:
```bash
mkdir backtranslation_output
for SHARD in $(seq -f "%02g" 0 24); do \
    fairseq-generate --fp16 \
        data-bin/wmt18_de_mono/shard${SHARD} \
        --path $CHECKPOINT_DIR/checkpoint_best.pt \
        --skip-invalid-size-inputs-valid-test \
        --max-tokens 4096 \
        --sampling --beam 1 \
    > backtranslation_output/sampling.shard${SHARD}.out; \
done
```

Extract and filter back-translations:
```bash
python examples/backtranslation/extract_bt_data.py \
    --minlen 1 --maxlen 250 --ratio 1.5 \
    --output backtranslation_output/bt_data --srclang en --tgtlang de \
    backtranslation_output/sampling.shard*.out
```

Combine parallel and BT data:
```bash
# Binarize BT data
TEXT=backtranslation_output
fairseq-preprocess \
    --source-lang en --target-lang de \
    --joined-dictionary \
    --srcdict data-bin/wmt18_en_de/dict.en.txt \
    --trainpref $TEXT/bt_data \
    --destdir data-bin/wmt18_en_de_bt \
    --workers 20

# Link parallel and BT data together
PARA_DATA=$(readlink -f data-bin/wmt18_en_de)
BT_DATA=$(readlink -f data-bin/wmt18_en_de_bt)
COMB_DATA=data-bin/wmt18_en_de_para_plus_bt
mkdir -p $COMB_DATA
# Create symlinks for dictionaries and data files
for LANG in en de; do \
    ln -s ${PARA_DATA}/dict.$LANG.txt ${COMB_DATA}/dict.$LANG.txt; \
    for EXT in bin idx; do \
        ln -s ${PARA_DATA}/train.en-de.$LANG.$EXT ${COMB_DATA}/train.en-de.$LANG.$EXT; \
        ln -s ${BT_DATA}/train.en-de.$LANG.$EXT ${COMB_DATA}/train1.en-de.$LANG.$EXT; \
        ln -s ${PARA_DATA}/valid.en-de.$LANG.$EXT ${COMB_DATA}/valid.en-de.$LANG.$EXT; \
        ln -s ${PARA_DATA}/test.en-de.$LANG.$EXT ${COMB_DATA}/test.en-de.$LANG.$EXT; \
    done; \
done
```

### 3. Train Final Model with Combined Data

```bash
CHECKPOINT_DIR=checkpoints_en_de_parallel_plus_bt
fairseq-train --fp16 \
    data-bin/wmt18_en_de_para_plus_bt \
    --upsample-primary 16 \
    --source-lang en --target-lang de \
    --arch transformer_wmt_en_de_big --share-all-embeddings \
    --dropout 0.3 --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 0.0007 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --max-tokens 3584 --update-freq 16 \
    --max-update 100000 \
    --save-dir $CHECKPOINT_DIR
```

Average and evaluate:
```bash
python scripts/average_checkpoints.py \
    --inputs $CHECKPOINT_DIR \
    --num-epoch-checkpoints 10 \
    --output $CHECKPOINT_DIR/checkpoint.avg10.pt

bash examples/backtranslation/sacrebleu.sh \
    wmt17 \
    en-de \
    data-bin/wmt18_en_de \
    data-bin/wmt18_en_de/code \
    $CHECKPOINT_DIR/checkpoint.avg10.pt
```

## Citation
```bibtex
@inproceedings{edunov2018backtranslation,
  title = {Understanding Back-Translation at Scale},
  author = {Edunov, Sergey and Ott, Myle and Auli, Michael and Grangier, David},
  booktitle = {Conference of the Association for Computational Linguistics (ACL)},
  year = 2018,
}
```