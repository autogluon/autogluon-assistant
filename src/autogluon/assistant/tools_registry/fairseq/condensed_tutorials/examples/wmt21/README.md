# Condensed: WMT 21

Summary: This tutorial introduces Facebook AI's WMT'21 neural machine translation models, featuring two primary implementations: a dense-24-wide model for translating any language to English, and another for English to any language translation. The tutorial provides download links for pre-trained models and references an eval.sh script for implementation guidance. These models can help with multilingual translation tasks, particularly for news content, and represent state-of-the-art translation capabilities as documented in Tran et al.'s 2021 paper. The key functionality is bidirectional translation between English and multiple other languages using dense transformer architectures.

*This is a condensed version that preserves essential implementation details and context.*

# WMT 21 Models

## Available Models

Facebook AI's WMT'21 news translation task submission [(Tran et al., 2021)](https://arxiv.org/abs/2108.03265) provides two main model types:

| Model | Description | Download |
|---|---|---|
| `wmt21.dense-24-wide.X-En` | Any language to English | [download (.tar.gz)](https://dl.fbaipublicfiles.com/fairseq/models/wmt21.dense-24-wide.X-En.tar.gz) |
| `wmt21.dense-24-wide.En-X` | English to any language | [download (.tar.gz)](https://dl.fbaipublicfiles.com/fairseq/models/wmt21.dense-24-wide.En-X.tar.gz) |

## Usage

Refer to the `eval.sh` script in the downloaded package for implementation details.

## Citation
```bibtex
@inproceedings{tran2021facebook
  title={Facebook AI's WMT21 News Translation Task Submission},
  author={Chau Tran and Shruti Bhosale and James Cross and Philipp Koehn and Sergey Edunov and Angela Fan},
  booktitle={Proc. of WMT},
  year={2021},
}
```