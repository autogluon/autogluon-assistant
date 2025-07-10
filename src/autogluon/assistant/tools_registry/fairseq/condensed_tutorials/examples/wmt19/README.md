# Condensed: WMT 19

Summary: This tutorial demonstrates how to use Facebook-FAIR's pre-trained WMT'19 translation models in PyTorch. It covers implementation of transformer-based neural machine translation between English, German, and Russian language pairs using torch.hub. Key functionalities include loading translation ensembles (combining 4 models), performing text translation, and using language models for text generation. The code shows how to handle tokenization with Moses and fastBPE preprocessing. The tutorial is valuable for implementing high-quality machine translation systems, text generation, and working with pre-trained transformer models in PyTorch.

*This is a condensed version that preserves essential implementation details and context.*

# WMT 19 Translation Models

## Pre-trained Models Overview

Facebook-FAIR's WMT'19 news translation task submission [(Ng et al., 2019)](https://arxiv.org/abs/1907.06616) provides several translation models:

- **Translation Ensembles**: En-De, De-En, En-Ru, Ru-En
- **Language Models**: English, German, Russian
- **Single Models (pre-finetuning)**: Available for all language pairs

## Implementation Requirements

```bash
pip install fastBPE sacremoses
```

## Usage with torch.hub

### Translation Example

```python
import torch

# English to German translation (ensemble of 4 models)
en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de', 
                      checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt',
                      tokenizer='moses', bpe='fastbpe')
en2de.translate("Machine learning is great!")  # 'Maschinelles Lernen ist großartig!'

# Similar pattern for other language pairs: de-en, en-ru, ru-en
```

### Language Modeling Example

```python
# Sample from the English LM
en_lm = torch.hub.load('pytorch/fairseq', 'transformer_lm.wmt19.en', 
                      tokenizer='moses', bpe='fastbpe')
en_lm.sample("Machine learning is")  # Generates continuation text

# Similar pattern for German and Russian LMs
```

## Key Implementation Details

- Models use transformer architecture
- Translation models are available as ensembles (combining 4 models)
- Single models before finetuning use FFN size of 8192
- All models use fastBPE tokenization and Moses preprocessing
- Language models can be used for text generation/completion

## Citation

```bibtex
@inproceedings{ng2019facebook},
  title = {Facebook FAIR's WMT19 News Translation Task Submission},
  author = {Ng, Nathan and Yee, Kyra and Baevski, Alexei and Ott, Myle and Auli, Michael and Edunov, Sergey},
  booktitle = {Proc. of WMT},
  year = 2019,
}
```