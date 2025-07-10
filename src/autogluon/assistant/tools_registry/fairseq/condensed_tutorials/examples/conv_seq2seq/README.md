# Condensed: Convolutional Sequence to Sequence Learning (Gehring et al., 2017)

Summary: This tutorial provides implementation details for Convolutional Sequence to Sequence Learning models (Gehring et al., 2017) in fairseq. It offers pre-trained models for neural machine translation on WMT14 English-French, WMT14 English-German, and WMT17 English-German datasets with download links and test sets. The tutorial helps with implementing convolutional sequence-to-sequence architectures for translation tasks using the specific model architectures `fconv_wmt_en_de` and `fconv_wmt_en_fr`. Key features include access to pre-trained models, implementation guidance, and references to reproduce results from the original paper.

*This is a condensed version that preserves essential implementation details and context.*

# Convolutional Sequence to Sequence Learning (Gehring et al., 2017)

## Pre-trained Models

| Description | Dataset | Model | Test set(s) |
|-------------|---------|-------|------------|
| Convolutional ([Gehring et al., 2017](https://arxiv.org/abs/1705.03122)) | [WMT14 English-French](http://statmt.org/wmt14/translation-task.html#Download) | [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/wmt14.v2.en-fr.fconv-py.tar.bz2) | [newstest2014](https://dl.fbaipublicfiles.com/fairseq/data/wmt14.v2.en-fr.newstest2014.tar.bz2), [newstest2012/2013](https://dl.fbaipublicfiles.com/fairseq/data/wmt14.v2.en-fr.ntst1213.tar.bz2) |
| Convolutional ([Gehring et al., 2017](https://arxiv.org/abs/1705.03122)) | [WMT14 English-German](http://statmt.org/wmt14/translation-task.html#Download) | [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/wmt14.en-de.fconv-py.tar.bz2) | [newstest2014](https://dl.fbaipublicfiles.com/fairseq/data/wmt14.en-de.newstest2014.tar.bz2) |
| Convolutional ([Gehring et al., 2017](https://arxiv.org/abs/1705.03122)) | [WMT17 English-German](http://statmt.org/wmt17/translation-task.html#Download) | [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/wmt17.v2.en-de.fconv-py.tar.bz2) | [newstest2014](https://dl.fbaipublicfiles.com/fairseq/data/wmt17.v2.en-de.newstest2014.tar.bz2) |

## Implementation Details

To reproduce results for WMT'14 En-De and WMT'14 En-Fr, use the `fconv_wmt_en_de` and `fconv_wmt_en_fr` model architectures as described in the [translation README](../translation/README.md).

## Citation

```bibtex
@inproceedings{gehring2017convs2s,
  title = {Convolutional Sequence to Sequence Learning},
  author = {Gehring, Jonas, and Auli, Michael and Grangier, David and Yarats, Denis and Dauphin, Yann N},
  booktitle = {Proc. of ICML},
  year = 2017,
}
```