# Condensed: Data card for the paper "Efficient Large Scale Language Modeling with Mixtures of Experts"

Summary: This tutorial details the dataset used to train a 1.1T parameter language model by FAIR. It provides implementation knowledge about large-scale language model training data composition (453GB, 112B tokens), covering specific preprocessing techniques and dataset proportions. The tutorial helps with tasks related to creating training datasets for large language models, including data filtering, validation set creation, and handling web-crawled content. Key features covered include the composition of six major data sources (BookCorpus, Wikipedia, CC-News, OpenWebText, CC-Stories, CC100), preprocessing approaches for web content, and considerations for handling potentially sensitive information in training data.

*This is a condensed version that preserves essential implementation details and context.*

# Data Card for 1.1T Parameter Model Training Dataset

## Overview
This data card describes the dataset used to train the 1.1T parameter language model as presented in "Efficient Large Scale Language Modeling with Mixtures of Experts".

## Dataset Composition
- **Total size**: 453 GB containing 112B tokens
- **Source datasets**:
  - BookCorpus: 10K+ unpublished books (4GB)
  - English Wikipedia: excluding lists, tables, headers (12GB)
  - CC-News: 63M English news articles (76GB)
  - OpenWebText: recreation of WebText used for GPT-2 (38GB)
  - CC-Stories: CommonCrawl subset filtered for story-like content (31GB)
  - English CC100: CommonCrawl snapshots filtered to match Wikipedia style (292GB)

## Implementation Details
- A validation set of ~150MB was held out from pretraining data
- Validation samples were drawn proportionally to each dataset's size in the corpus
- The dataset is self-contained and doesn't rely on external resources

## Data Characteristics
- Instances consist of raw text documents
- No explicit labels or targets associated with instances
- No explicit relationships between individual instances

## Important Considerations
- CC100 and CC-Stories are filtered subsets of CommonCrawl
- Common Crawl portions may contain potentially offensive or sensitive content
- May include identifiable information about individuals, especially public figures
- Dataset may contain sensitive information (racial, political, religious, etc.) from web sources

## Creation Purpose
Created by FAIR (Fundamental Artificial Intelligence Research) specifically for pre-training the 1.1T parameter language model.

# Collection Process and Data Handling

## Data Collection

- The dataset is a union of six publicly available datasets
- Different components were collected over different timeframes:
  - CC-News: English news articles from September 2016 to February 2019
  - English CC-100: Extracted from CommonCrawl snapshots between January-December 2018
- Data was mined, filtered, and sampled by machines
- No ethical review processes were conducted

## Preprocessing and Cleaning

- Standard cleaning and reformatting was applied:
  - Removal of repetitive/non-informative text (e.g., "Chapter One")
  - Removal of boilerplate content (e.g., "This ebook by Project Gutenberg")
- The preprocessing software is proprietary to Meta Platforms and not publicly available
- Raw component datasets remain available at their original sources

## Data Usage

- Primary use: Pre-training the language models described in the paper
- Potential applications: Pre-training English language models for various language tasks
- The pipeline establishes a scalable infrastructure for mining datasets for large-scale model training

```
Important note: The dataset itself will not be distributed to third parties 
outside of the entity on behalf of which it was created.
```

## Distribution Information

- The dataset will not be distributed
- No DOI has been assigned
- No licensing or terms of use are specified as the dataset is not being distributed
- No third-party IP restrictions or export controls apply to the dataset

# FAIR Pile Dataset Maintenance Information

## Dataset Support and Maintenance

- **Maintained by**: FAIR (Fundamental Artificial Intelligence Research)
- **Contact**: Refer to the main document
- **Erratum**: N/A
- **Updates**: No plans for updating the dataset
- **Older versions**: N/A - No specific plans for maintaining older versions
- **Extensions/Contributions**: No formal mechanism for external contributions

## People-Related Data Considerations

- Not applicable to this dataset

## References

```
Yinhan Liu, Myle Ott, Naman Goyal, et al. 2019. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692.

Yukun Zhu, Ryan Kiros, Richard Zemel, et al. 2019. Aligning books and movies: Towards story-like visual explanations by watching movies and reading books. arXiv:1506.06724.

Sebastian Nagel. 2016. Cc-news. http://web.archive.org/save/http://commoncrawl.org/2016/10/news-dataset-available.

Aaron Gokaslan and Vanya Cohen. 2019. Openwebtext corpus. http://web.archive.org/save/http://Skylion007.github.io/OpenWebTextCorpus

Trieu H Trinh and Quoc V Le. 2018. A simple method for commonsense reasoning. arXiv preprint arXiv:1806.02847.

Guillaume Wenzek, et al. 2020. CCNet: Extracting high quality monolingual datasets from web crawl data. In Proceedings of the 12th LREC, pages 4003–4012.
```