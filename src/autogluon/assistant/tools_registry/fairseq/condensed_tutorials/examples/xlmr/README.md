# Condensed: Unsupervised Cross-lingual Representation Learning at Scale (XLM-RoBERTa)

Summary: This tutorial introduces XLM-RoBERTa, a cross-lingual sentence encoder supporting 100 languages with models ranging from 250M to 10.7B parameters. It demonstrates implementation techniques for loading XLM-R models via torch.hub or from local files, encoding/decoding text across multiple languages using SentencePiece tokenization, and extracting features from specific or all layers. The tutorial helps with multilingual NLP tasks including text encoding, feature extraction for downstream tasks, and cross-lingual representation learning. Key functionalities covered include model initialization, multilingual text processing, and hidden state extraction for transfer learning applications.

*This is a condensed version that preserves essential implementation details and context.*

# XLM-RoBERTa: Unsupervised Cross-lingual Representation Learning at Scale

## Overview
XLM-R is a state-of-the-art cross-lingual sentence encoder trained on 2.5T of filtered CommonCrawl data in 100 languages. The model family includes base, large, XL, and XXL variants.

## Available Models

| Model | Architecture | Parameters | Vocab Size | Download |
|-------|-------------|------------|------------|----------|
| `xlmr.base` | BERT-base | 250M | 250k | [xlm.base.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/xlmr.base.tar.gz) |
| `xlmr.large` | BERT-large | 560M | 250k | [xlm.large.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/xlmr.large.tar.gz) |
| `xlmr.xl` | 36 layers, 2560 dim | 3.5B | 250k | [xlm.xl.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/xlmr/xlmr.xl.tar.gz) |
| `xlmr.xxl` | 48 layers, 4096 dim | 10.7B | 250k | [xlm.xxl.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/xlmr/xlmr.xxl.tar.gz) |

## Implementation Examples

### Loading XLM-R with torch.hub (PyTorch >= 1.1)
```python
import torch
xlmr = torch.hub.load('pytorch/fairseq:main', 'xlmr.large')
xlmr.eval()  # disable dropout (or leave in train mode to finetune)
```

### Loading XLM-R for PyTorch 1.0 or custom models
```python
# Download and extract model
wget https://dl.fbaipublicfiles.com/fairseq/models/xlmr.large.tar.gz
tar -xzvf xlmr.large.tar.gz

# Load the model in fairseq
from fairseq.models.roberta import XLMRModel
xlmr = XLMRModel.from_pretrained('/path/to/xlmr.large', checkpoint_file='model.pt')
xlmr.eval()  # disable dropout (or leave in train mode to finetune)
```

### Applying SentencePiece Model (SPM) encoding
```python
# English
en_tokens = xlmr.encode('Hello world!')
assert en_tokens.tolist() == [0, 35378, 8999, 38, 2]
xlmr.decode(en_tokens)  # 'Hello world!'

# Chinese
zh_tokens = xlmr.encode('你好，世界')
assert zh_tokens.tolist() == [0, 6, 124084, 4, 3221, 2]
xlmr.decode(zh_tokens)  # '你好，世界'

# Hindi
hi_tokens = xlmr.encode('नमस्ते दुनिया')
assert hi_tokens.tolist() == [0, 68700, 97883, 29405, 2]
xlmr.decode(hi_tokens)  # 'नमस्ते दुनिया'
```

### Feature Extraction
```python
# Extract the last layer's features
last_layer_features = xlmr.extract_features(zh_tokens)
assert last_layer_features.size() == torch.Size([1, 6, 1024])

# Extract all layer's features (layer 0 is the embedding layer)
all_layers = xlmr.extract_features(zh_tokens, return_all_hiddens=True)
assert len(all_layers) == 25
assert torch.all(all_layers[-1] == last_layer_features)
```

## Performance Benchmarks

XLM-R achieves state-of-the-art results on cross-lingual understanding tasks:

- **XNLI**: XLM-R XXL achieves 86.0% average accuracy across 15 languages
- **MLQA**: XLM-R XXL achieves 74.8/56.6 F1/EM scores across 7 languages

For detailed performance metrics, refer to the original papers.