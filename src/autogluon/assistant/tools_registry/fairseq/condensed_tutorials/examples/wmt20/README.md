# Condensed: WMT 20

Summary: This tutorial provides access to Facebook-FAIR's WMT'20 neural machine translation models, featuring transformer-based translation systems for Tamil-English and Inuktitut-English language pairs. It demonstrates how to implement these pre-trained models using torch.hub for both translation tasks and language modeling. Developers can use code examples to load models, translate text between languages, and generate text completions with language models. Key functionalities include bidirectional translation between English and low-resource languages (Tamil, Inuktitut), with separate models for news and Nunavut Hansard domains for Inuktitut.

*This is a condensed version that preserves essential implementation details and context.*

# WMT 20 Models

This page provides access to Facebook-FAIR's WMT'20 news translation task submission models [(Chen et al., 2020)](https://arxiv.org/abs/2011.08298).

## Available Models

### Translation Models
- Tamil-English (`transformer.wmt20.ta-en`) - [download](https://dl.fbaipublicfiles.com/fairseq/models/wmt20.ta-en.single.tar.gz)
- English-Tamil (`transformer.wmt20.en-ta`) - [download](https://dl.fbaipublicfiles.com/fairseq/models/wmt20.en-ta.single.tar.gz)
- Inuktitut-English News (`transformer.wmt20.iu-en.news`) - [download](https://dl.fbaipublicfiles.com/fairseq/models/wmt20.iu-en.news.single.tar.gz)
- English-Inuktitut News (`transformer.wmt20.en-iu.news`) - [download](https://dl.fbaipublicfiles.com/fairseq/models/wmt20.en-iu.news.single.tar.gz)
- Inuktitut-English NH (`transformer.wmt20.iu-en.nh`) - [download](https://dl.fbaipublicfiles.com/fairseq/models/wmt20.iu-en.nh.single.tar.gz)
- English-Inuktitut NH (`transformer.wmt20.en-iu.nh`) - [download](https://dl.fbaipublicfiles.com/fairseq/models/wmt20.en-iu.nh.single.tar.gz)

### Language Models
- English LM (`transformer_lm.wmt20.en`) - [download](https://dl.fbaipublicfiles.com/fairseq/models/wmt20.en.tar.gz)
- Tamil LM (`transformer_lm.wmt20.ta`) - [download](https://dl.fbaipublicfiles.com/fairseq/models/wmt20.ta.tar.gz)
- Inuktitut News LM (`transformer_lm.wmt20.iu.news`) - [download](https://dl.fbaipublicfiles.com/fairseq/models/wmt20.iu.news.tar.gz)
- Inuktitut NH LM (`transformer_lm.wmt20.iu.nh`) - [download](https://dl.fbaipublicfiles.com/fairseq/models/wmt20.iu.nh.tar.gz)

## Implementation Examples (torch.hub)

### Translation Usage

```python
import torch

# English to Tamil
en2ta = torch.hub.load('pytorch/fairseq', 'transformer.wmt20.en-ta')
en2ta.translate("Machine learning is great!")  # 'இயந்திரக் கற்றல் அருமை!'

# Tamil to English
ta2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt20.ta-en')
ta2en.translate("இயந்திரக் கற்றல் அருமை!")  # 'Machine learning is great!'

# English to Inuktitut
en2iu = torch.hub.load('pytorch/fairseq', 'transformer.wmt20.en-iu.news')
en2iu.translate("machine learning is great!")  # 'ᖃᒧᑕᐅᔭᓄᑦ ᐃᓕᓐᓂᐊᕐᓂᖅ ᐱᐅᔪᒻᒪᕆᒃ!'

# Inuktitut to English
iu2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt20.iu-en.news')
iu2en.translate("ᖃᒧᑕᐅᔭᓄᑦ ᐃᓕᓐᓂᐊᕐᓂᖅ ᐱᐅᔪᒻᒪᕆᒃ!")  # 'Machine learning excellence!'
```

### Language Model Usage

```python
# English LM
en_lm = torch.hub.load('pytorch/fairseq', 'transformer_lm.wmt20.en')
en_lm.sample("Machine learning is")  # 'Machine learning is a type of artificial intelligence...'

# Tamil LM
ta_lm = torch.hub.load('pytorch/fairseq', 'transformer_lm.wmt20.ta')
ta_lm.sample("இயந்திரக் கற்றல் என்பது செயற்கை நுண்ணறிவின்")  # 'இயந்திரக் கற்றல் என்பது செயற்கை நுண்ணறிவின் ஒரு பகுதியாகும்.'

# Inuktitut LM
iu_lm = torch.hub.load('pytorch/fairseq', 'transformer_lm.wmt20.iu.news')
iu_lm.sample("ᖃᒧᑕᐅᔭᓄᑦ ᐃᓕᓐᓂᐊᕐᓂᖅ")  # 'ᖃᒧᑕᐅᔭᓄᑦ ᐃᓕᓐᓂᐊᕐᓂᖅ, ᐊᒻᒪᓗ ᓯᓚᐅᑉ ᐊᓯᙳᖅᐸᓪᓕᐊᓂᖓᓄᑦ...'
```

## Citation
```bibtex
@inproceedings{chen2020facebook
  title={Facebook AI's WMT20 News Translation Task Submission},
  author={Peng-Jen Chen and Ann Lee and Changhan Wang and Naman Goyal and Angela Fan and Mary Williamson and Jiatao Gu},
  booktitle={Proc. of WMT},
  year={2020},
}
```