# Condensed: MMS Zero-shot Speech Recognition

Summary: This tutorial explains how to implement zero-shot multilingual speech recognition using the MMS model, which works with 1000+ languages through uroman text representation. It covers: (1) creating uroman-based lexicons and optional N-gram language models with KenLM, (2) running inference with either lexicon-only or N-gram LM approaches using provided Python commands, and (3) tuning key parameters like word scores and language model weights. The tutorial helps with speech recognition tasks for low-resource languages by providing complete implementation code, parameter recommendations, and model download links for a pre-trained system that only requires language-specific lexicons at inference time.

*This is a condensed version that preserves essential implementation details and context.*

# MMS Zero-shot Speech Recognition

A multilingual speech recognition model for nearly all world languages using uroman text as intermediate representation. Pre-trained on 1000+ languages, it only requires lexicon and optional N-gram language models for unseen languages at inference time.

**Resources:**
- [Model download](https://dl.fbaipublicfiles.com/mms/zeroshot/model.pt)
- [Dictionary download](https://dl.fbaipublicfiles.com/mms/zeroshot/tokens.txt)
- [Demo](https://huggingface.co/spaces/mms-meta/mms-zeroshot)

## Inference Setup

### 1. Prepare uroman-based lexicon
Create a lexicon file using [uroman](https://github.com/isi-nlp/uroman) with this format:
```
abiikira a b i i k i r a |
úwangaba u w a n g a b a |
banakana b a n a k a n a |
```
**Important:** Each uroman token must appear in the model's token dictionary.

### 2. Optional: Prepare N-gram language model
Build language models with [KenLM](https://github.com/kpu/kenlm). Even 1-gram LMs can produce good results.

## Inference Commands

### Lexicon-only inference
```bash
PYTHONPATH=. PREFIX=INFER HYDRA_FULL_ERROR=1 python examples/speech_recognition/new/infer.py -m \
    --config-dir examples/mms/asr/config/ \
    --config-name infer_common decoding.type=kenlm \
    dataset.max_tokens=2000000 distributed_training.distributed_world_size=1 \
    "common_eval.path=${model_path}" task.data=${data_path} \
    dataset.gen_subset=mms_eng:${subset} decoding.lexicon=${lex_filepath} \
    decoding.lmpath=${lm_filepath} decoding.lmweight=0 decoding.wordscore=${wrdscore} \
    decoding.silweight=0 decoding.results_path=${res_path} \
    decoding.beam=${beam}
```

### N-gram LM inference
```bash
PYTHONPATH=. PREFIX=INFER HYDRA_FULL_ERROR=1 python examples/speech_recognition/new/infer.py -m \
    --config-dir examples/mms/asr/config/ \
    --config-name infer_common decoding.type=kenlm \
    dataset.max_tokens=2000000 distributed_training.distributed_world_size=1 \
    "common_eval.path=${model_path}" task.data=${data_path} \
    dataset.gen_subset=mms_eng:${subset} decoding.lexicon=${lex_filepath} \
    decoding.lmpath=${lm_filepath} decoding.lmweight=${lmweight} decoding.wordscore=${wrdscore} \
    decoding.silweight=0 decoding.results_path=${res_path} \
    decoding.beam=${bs}
```

**Key parameters:**
- `wrdscore`: -3.5 (lexicon-only) or -0.18 (with LM) - can be tuned
- `lmweight`: 1.48 (with LM) - can be tuned with wrdscore
- `bs`: 2000 or 500 (batch size)

**Note:** Commands won't calculate CER directly if your script isn't in the dictionary. You'll need to calculate CER manually after generation.

## License
CC-BY-NC 4.0

## Citation
```
@article{zhao2024zeroshot,
  title={Scaling a Simple Approach to Zero-shot Speech Recognition},
  author={Jinming Zhao, Vineel Pratap and Michael Auli},
  journal={arXiv},
  year={2024}
}
```