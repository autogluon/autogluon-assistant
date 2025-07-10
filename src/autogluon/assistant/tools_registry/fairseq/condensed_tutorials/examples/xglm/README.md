# Condensed: Few-shot Learning with Multilingual Language Models

Summary: This tutorial introduces XGLM, a family of multilingual generative language models for few-shot learning across 30-134 languages. It demonstrates how to load pre-trained XGLM models (ranging from 564M to 7.5B parameters) using the fairseq library, handle multilingual text preprocessing, and implement few-shot evaluation for tasks like COPA (Choice of Plausible Alternatives). The tutorial covers techniques for calculating log probabilities of text sequences, comparing alternative completions, and working with the XStoryCloze dataset—a multilingual benchmark for evaluating language models across 10 languages. Key functionalities include text scoring, token handling, and cross-lingual inference.

*This is a condensed version that preserves essential implementation details and context.*

# Few-shot Learning with Multilingual Language Models

## Introduction

XGLM is a family of multilingual generative language models trained on a balanced corpus covering diverse languages. The largest model (7.5B parameters) outperforms GPT-3 of comparable size in multilingual tasks, including commonsense reasoning and natural language inference.

## Pre-trained Models

| Model | Layers | Model Dim | FFN Dim | Languages | Download |
|---|---|---|---|---|---|
| `XGLM 564M` | 24 | 1024 | 4096 | 30 languages | [xglm.564M.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/xglm/xglm.564M.tar.gz) |
| `XGLM 1.7B` | 24 | 2048 | 8192 | 30 languages | [xglm.1.7B.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/xglm/xglm.1.7B.tar.gz) |
| `XGLM 2.9B` | 48 | 2048 | 8192 | 30 languages | [xglm.2.9B.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/xglm/xglm.2.9B.tar.gz) |
| `XGLM 7.5B` | 32 | 4096 | 16384 | 30 languages | [xglm.7.5B.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/xglm/xglm.7.5B.tar.gz) |
| `XGLM 4.5B` | 48 | 2048 | 16384 | 134 languages | [xglm.4.5B.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/xglm/xglm.4.5B.tar.gz) |

## Pre-training Data Format

Models were pre-trained with paragraphs separated by newlines and documents separated by double newlines. During preprocessing, newlines are replaced with end-of-sentence symbols (`</s>`).

```python
from fairseq.models.transformer_lm import TransformerLanguageModel

model_dir = 'path_to_decompressed_tar_gz_dir'
lm = TransformerLanguageModel.from_pretrained(model_dir, bpe='sentencepiece')

text = """First paragraph of the first document.
Second paragraph of the first document.

First paragraph of the second document.
"""
tokens = lm.score(text, replace_newlines_with_eos=True)['tokens']
assert '\n' not in lm.decode(tokens)  # no newlines were encoded
```

## Evaluation Example (COPA)

Example of evaluating on the Choice of Plausible Alternatives task in multiple languages:

```python
from fairseq.models.transformer_lm import TransformerLanguageModel

model_dir = 'path_to_decompressed_tar_gz_dir'
lm = TransformerLanguageModel.from_pretrained(model_dir, bpe='sentencepiece')
lm = lm.eval().half().cuda()

def get_logprobs(prompt):
    import re
    prompt = re.sub('\n+', '\n', prompt)  # collapse repeated newlines
    return lm.score(prompt, replace_newlines_with_eos=True)['positional_scores']
    
def COPA_eval(prompt, alternative1, alternative2):
    lprob1 = get_logprobs(prompt + "\n" + alternative1).sum()
    lprob2 = get_logprobs(prompt + "\n" + alternative2).sum()
    return 0 if lprob1 > lprob2 else 1
```

## XStoryCloze Dataset

XStoryCloze is a multilingual dataset for few-shot evaluation, consisting of professional translations of the English StoryCloze validation split to 10 languages. Available under CC BY-SA 4.0 license.

Download: [xstorycloze.zip](https://dl.fbaipublicfiles.com/xstorycloze.zip)

| Language | ar | es | eu | hi | id | my | ru | sw | te | zh |
|---|---|---|---|---|---|---|---|---|---|---|
| Train size | 360 | 360 | 360 | 360 | 360 | 360 | 360 | 360 | 360 | 360 |
| Eval size | 1511 | 1511 | 1511 | 1511 | 1511 | 1511 | 1511 | 1511 | 1511 | 1511 |