# Condensed: Wikipedia Biographies of Women

Summary: This tutorial provides implementation guidance for creating and using a Wikipedia biographies dataset focused on women. It covers techniques for training dataset creation using WikiSum and CommonCrawl, with specific filtering for biographical articles through WikiData occupation queries. The tutorial helps with downloading and structuring evaluation datasets containing full Wikipedia articles across different categories of women, collecting web evidence through search engines, and implementing section-by-section text generation using regex patterns. Key functionalities include handling multiple occupations per person, extracting article sections, and integrating web evidence while excluding Wikipedia sources from search results.

*This is a condensed version that preserves essential implementation details and context.*

# Wikipedia Biographies of Women - Implementation Guide

## Training Dataset Creation

The training dataset is based on WikiSum from the paper [Generating Wikipedia by Summarizing Long Sequences](https://arxiv.org/pdf/1801.10198.pdf).

**Implementation details:**
- Dataset must be generated following instructions in this [Github Repository](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/data_generators/wikisum)
- Uses CommonCrawl version (static, open source web crawl)
- Requires Google Cloud with potentially increased resource limits
- Biography filtering: Uses WikiData occupation queries to identify biographical articles

> **Warning:** Lower coverage in training data may impair model's ability to retrieve information and generate verifiable content.

## Evaluation Dataset

**Download commands:**
```bash
wget -N 'https://dl.fbaipublicfiles.com/fairseq/womenbios_dataset.zip'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'
```

**Dataset structure:**
- Full text Wikipedia articles in four categories:
  - Women in Africa
  - Women in Asia
  - Women in Science
  - Women (general)
- Includes Wikipedia article URLs
- Occupation data from WikiData
- Web evidence from search queries

**Web evidence collection:**
- Uses search engine from [Internet-Augmented Dialogue Generation](https://arxiv.org/abs/2107.07566)
- Wikipedia sources are excluded from search results

## Section-by-Section Generation

Wikipedia articles are structured with sections separated by headings. Extract sections using regex:

```python
section_header_re = re.compile(r"(?<!=)==([^=]+)==(?!=)")
```

## Implementation Notes

- Multiple occupations per person are preserved from WikiData
- Potential improvements:
  - Larger generative pre-trained models
  - Larger-scale retrieval
  - Specialized retrieval encoder for Wikipedia/biographies
  - Parameter tuning for training and generation

## License

CC-BY-NC, with Wikipedia-sourced text under CC-BY-SA.