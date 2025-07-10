# Condensed: CamemBERT: a Tasty French Language Model

Summary: This tutorial introduces CamemBERT, a French language model based on RoBERTa architecture, with various pre-trained versions ranging from 110M to 335M parameters. It demonstrates implementation techniques for loading CamemBERT models via PyTorch's torch.hub or from local files, and showcases key functionalities including masked language prediction (fill_mask) and feature extraction from different layers. The tutorial helps with coding tasks related to French NLP, including model initialization, text prediction, and extracting embeddings for downstream tasks, making it valuable for developers working on French language processing applications.

*This is a condensed version that preserves essential implementation details and context.*

# CamemBERT: a Tasty French Language Model

## Introduction

CamemBERT is a pretrained language model trained on 138GB of French text based on RoBERTa architecture.

## Pre-trained Models

| Model | #params | Training data | Architecture |
|-------|---------|---------------|-------------|
| `camembert-base` | 110M | OSCAR (138 GB) | Base |
| `camembert-large` | 335M | CCNet (135 GB) | Large |
| `camembert-base-ccnet` | 110M | CCNet (135 GB) | Base |
| `camembert-base-wikipedia-4gb` | 110M | Wikipedia (4 GB) | Base |
| `camembert-base-oscar-4gb` | 110M | OSCAR subsample (4 GB) | Base |
| `camembert-base-ccnet-4gb` | 110M | CCNet subsample (4 GB) | Base |

## Implementation Examples

### Loading from torch.hub (PyTorch >= 1.1)
```python
import torch
camembert = torch.hub.load('pytorch/fairseq', 'camembert')
camembert.eval()  # disable dropout
```

### Loading for PyTorch 1.0 or custom models
```python
# Download and extract model
wget https://dl.fbaipublicfiles.com/fairseq/models/camembert-base.tar.gz
tar -xzvf camembert.tar.gz

# Load the model
from fairseq.models.roberta import CamembertModel
camembert = CamembertModel.from_pretrained('/path/to/camembert')
camembert.eval()
```

### Filling Masks
```python
masked_line = 'Le camembert est <mask> :)'
camembert.fill_mask(masked_line, topk=3)
# [('Le camembert est délicieux :)', 0.4909118115901947, ' délicieux'),
#  ('Le camembert est excellent :)', 0.10556942224502563, ' excellent'),
#  ('Le camembert est succulent :)', 0.03453322499990463, ' succulent')]
```

### Feature Extraction
```python
# Extract last layer features
line = "J'aime le camembert !"
tokens = camembert.encode(line)
last_layer_features = camembert.extract_features(tokens)
assert last_layer_features.size() == torch.Size([1, 10, 768])

# Extract all layers' features
all_layers = camembert.extract_features(tokens, return_all_hiddens=True)
assert len(all_layers) == 13
assert torch.all(all_layers[-1] == last_layer_features)
```