# Condensed: XStoryCloze consists of professional translation of the validation split of the [English StoryCloze dataset](https://cs.rochester.edu/nlp/rocstories/) (Spring 2016 version) to 10 other languages. This dataset is released by FAIR (Fundamental Artificial Intelligence Research) alongside the paper [Few-shot Learning with Multilingual Generative Language Models. EMNLP 2022](https://arxiv.org/abs/2112.10668).

Summary: This tutorial introduces XStoryCloze, a multilingual dataset translating the English StoryCloze validation split into 10 languages (including Russian, Chinese, Spanish, and others). It helps with implementing multilingual story completion tasks and evaluating language models' zero/few-shot learning capabilities across languages. The tutorial covers how to access and process the dataset, create aligned training (360 examples) and test (1510 examples) splits, and maintain proper licensing (CC BY-SA 4.0). This resource is particularly valuable for developing and testing multilingual NLP models that perform narrative understanding and completion tasks.

*This is a condensed version that preserves essential implementation details and context.*

# XStoryCloze Dataset

A multilingual translation of the English StoryCloze validation split (Spring 2016) into 10 languages, released by FAIR for evaluating zero/few-shot learning capabilities of multilingual language models.

## Key Details

- **Languages**: ru, zh (Simplified), es (Latin America), ar, hi, id, te, sw, eu, my
- **Data Splits**: 360 training examples, 1510 test examples per language
- **Alignment**: All language files maintain line-by-line alignment

## Accessing English StoryCloze

Request the original dataset from the [official website](https://cs.rochester.edu/nlp/rocstories/), then create splits matching XStoryCloze:

```bash
# Create training split (first 361 lines)
head -361 spring2016.val.tsv > spring2016.val.en.tsv.split_20_80_train.tsv

# Create evaluation split (header + last 1511 lines)
head -1 spring2016.val.tsv > spring2016.val.en.tsv.split_20_80_eval.tsv
tail -1511 spring2016.val.tsv >> spring2016.val.en.tsv.split_20_80_eval.tsv
```

## License

XStoryCloze is available under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/legalcode), the same license as the original English StoryCloze.

## Citation

```
@article{DBLP:journals/corr/abs-2112-10668,
  author    = {Xi Victoria Lin and Todor Mihaylov and Mikel Artetxe and Tianlu Wang and 
               Shuohui Chen and Daniel Simig and Myle Ott and Naman Goyal and 
               Shruti Bhosale and Jingfei Du and Ramakanth Pasunuru and Sam Shleifer and 
               Punit Singh Koura and Vishrav Chaudhary and Brian O'Horo and Jeff Wang and 
               Luke Zettlemoyer and Zornitsa Kozareva and Mona T. Diab and 
               Veselin Stoyanov and Xian Li},
  title     = {Few-shot Learning with Multilingual Language Models},
  journal   = {CoRR},
  volume    = {abs/2112.10668},
  year      = {2021},
  url       = {https://arxiv.org/abs/2112.10668}
}
```