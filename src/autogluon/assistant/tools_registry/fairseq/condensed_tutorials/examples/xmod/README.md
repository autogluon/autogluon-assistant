# Condensed: X-MOD: Lifting the Curse of Multilinguality by Pre-training Modular Transformers

Summary: This tutorial demonstrates X-MOD, a multilingual transformer model with language-specific modules. It covers implementation of pre-trained models ranging from BERT-base to BERT-large supporting 13-81 languages. The tutorial provides code for fine-tuning on Natural Language Inference tasks and running inference across multiple languages. Key functionalities include downloading pre-trained models, preprocessing MNLI data, fine-tuning with language-specific adapters, and performing cross-lingual inference where specifying the target language ID is crucial. This knowledge helps with implementing multilingual NLP models that maintain language-specific representations while enabling effective cross-lingual transfer.

*This is a condensed version that preserves essential implementation details and context.*

# X-MOD: Lifting the Curse of Multilinguality by Pre-training Modular Transformers

## Overview
X-MOD extends multilingual masked language models by incorporating language-specific modular components at each transformer layer. Each module serves only one language, and during cross-lingual transfer, these components are frozen and replaced with the target language module.

## Pre-trained Models

Model sizes range from BERT-base to BERT-large with varying training steps (125k-1M) and language support (13-81 languages). Key models include:
- `xmod.base.81.1M` (BERT-base, 1M steps, 81 languages)
- `xmod.large.prenorm.81.500k` (BERT-large, 500k steps, 81 languages)

## Fine-tuning on NLI

### 1) Download pre-trained model
```bash
MODEL=xmod.base.81.1M
wget https://dl.fbaipublicfiles.com/fairseq/models/xmod/$MODEL.tar.gz
tar -xzf $MODEL.tar.gz
```

### 2) Preprocess MNLI data
```bash
wget https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip
unzip multinli_1.0.zip
python ./examples/xmod/preprocess_nli.py \
    --sentencepiece-model $MODEL/sentencepiece.bpe.model \
    --train multinli_1.0/multinli_1.0_train.jsonl \
    --valid multinli_1.0/multinli_1.0_dev_matched.jsonl \
    --destdir multinli_1.0/fairseq
```

### 3) Fine-tune on MNLI
```bash
MAX_EPOCH=5
LR=1e-05
BATCH_SIZE=32
DATA_DIR=multinli_1.0/fairseq/bin

CUDA_VISIBLE_DEVICES=0 fairseq-train $DATA_DIR \
    --restore-file $MODEL/model.pt \
    --save-dir $MODEL/nli \
    --reset-optimizer --reset-dataloader --reset-meters \
    --best-checkpoint-metric accuracy \
    --maximize-best-checkpoint-metric \
    --task sentence_prediction_adapters \
    --num-classes 3 \
    --init-token 0 \
    --separator-token 2 \
    --max-positions 512 \
    --shorten-method "truncate" \
    --arch xmod_base \
    --dropout 0.1 \
    --attention-dropout 0.1 \
    --weight-decay 0.01 \
    --criterion sentence_prediction_adapters \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler fixed \
    --lr $LR \
    --fp16 \
    --fp16-init-scale 4 \
    --threshold-loss-scale 1 \
    --fp16-scale-window 128 \
    --batch-size $BATCH_SIZE \
    --required-batch-size-multiple 1 \
    --update-freq 1 \
    --max-epoch $MAX_EPOCH
```

### 4) Run inference in multiple languages
```python
from fairseq.models.xmod import XMODModel

MODEL='xmod.base.81.1M/nli'
DATA='multinli_1.0/fairseq/bin'

# Load model
model = XMODModel.from_pretrained(
            model_name_or_path=MODEL,
            checkpoint_file='checkpoint_best.pt', 
            data_name_or_path=DATA, 
            suffix='', 
            criterion='cross_entropy', 
            bpe='sentencepiece',  
            sentencepiece_model=DATA+'/input0/sentencepiece.bpe.model')
model = model.eval()  # disable dropout
model = model.half()  # use FP16
model = model.cuda()  # move to GPU

def predict(premise, hypothesis, lang):
    tokens = model.encode(premise, hypothesis)
    idx = model.predict('sentence_classification_head', tokens, lang_id=[lang]).argmax().item()
    dictionary = model.task.label_dictionary
    return dictionary[idx + dictionary.nspecial]

# Example predictions in different languages
predict(
    premise='X-Mod hat spezifische Module die für jede Sprache existieren.',
    hypothesis='X-Mod hat Module.',
    lang='de_DE'
)  # entailment

predict(
    premise='Londres es la capital del Reino Unido.',
    hypothesis='Londres está en Francia.',
    lang='es_XX',
)  # contradiction
```

**Key Implementation Note**: When running inference in non-English languages, you must explicitly specify the target language ID to use the appropriate language module.