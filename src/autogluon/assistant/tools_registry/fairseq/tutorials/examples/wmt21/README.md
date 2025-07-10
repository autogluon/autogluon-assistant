Summary: This tutorial introduces Facebook AI's WMT'21 neural machine translation models, featuring two primary implementations: a dense-24-wide model for translating any language to English, and another for English to any language translation. The tutorial provides download links for pre-trained models and references an eval.sh script for implementation guidance. These models can help with multilingual translation tasks, particularly for news content, and represent state-of-the-art translation capabilities as documented in Tran et al.'s 2021 paper. The key functionality is bidirectional translation between English and multiple other languages using dense transformer architectures.

# WMT 21

This page provides pointers to the models of Facebook AI's WMT'21 news translation task submission [(Tran et al., 2021)](https://arxiv.org/abs/2108.03265).

## Single best dense models

Model | Description | Download
---|---|---
`wmt21.dense-24-wide.X-En` | X-En | [download (.tar.gz)](https://dl.fbaipublicfiles.com/fairseq/models/wmt21.dense-24-wide.X-En.tar.gz)
`wmt21.dense-24-wide.En-X` | En-X | [download (.tar.gz)](https://dl.fbaipublicfiles.com/fairseq/models/wmt21.dense-24-wide.En-X.tar.gz)

## Example usage

See eval.sh


## Citation
```bibtex
@inproceedings{tran2021facebook
  title={Facebook AI’s WMT21 News Translation Task Submission},
  author={Chau Tran and Shruti Bhosale and James Cross and Philipp Koehn and Sergey Edunov and Angela Fan},
  booktitle={Proc. of WMT},
  year={2021},
}
```
