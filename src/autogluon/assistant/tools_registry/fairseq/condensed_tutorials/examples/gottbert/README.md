# Condensed: GottBERT: a pure German language model

Summary: This tutorial introduces GottBERT, a RoBERTa-based language model trained on 145GB of German text. It demonstrates how to implement the model using PyTorch, covering loading methods via torch.hub or custom installation. The tutorial showcases key functionalities including mask filling for predicting missing words in German sentences and feature extraction capabilities for obtaining embeddings from either the last layer or all layers. This resource is particularly valuable for developers working on German NLP tasks who need pre-trained language model implementations with examples of practical text processing applications.

*This is a condensed version that preserves essential implementation details and context.*

# GottBERT: A Pure German Language Model

## Implementation Details

GottBERT is a RoBERTa-based language model trained on 145GB of German text.

## Usage Examples

### Loading the Model

```python
# From torch.hub (PyTorch >= 1.1)
import torch
gottbert = torch.hub.load('pytorch/fairseq', 'gottbert-base')
gottbert.eval()  # disable dropout

# For PyTorch 1.0 or custom models
# 1. Download and extract
wget https://dl.gottbert.de/fairseq/models/gottbert-base.tar.gz
tar -xzvf gottbert.tar.gz

# 2. Load with fairseq
from fairseq.models.roberta import GottbertModel
gottbert = GottbertModel.from_pretrained('/path/to/gottbert')
gottbert.eval()
```

### Mask Filling

```python
masked_line = 'Gott ist <mask> ! :)'
gottbert.fill_mask(masked_line, topk=3)
# [('Gott ist gut ! :)',        0.3642110526561737,   ' gut'),
#  ('Gott ist überall ! :)',    0.06009674072265625,  ' überall'),
#  ('Gott ist großartig ! :)',  0.0370681993663311,   ' großartig')]
```

### Feature Extraction

```python
# Extract last layer features
line = "Der erste Schluck aus dem Becher der Naturwissenschaft macht atheistisch , aber auf dem Grunde des Bechers wartet Gott !"
tokens = gottbert.encode(line)
last_layer_features = gottbert.extract_features(tokens)
assert last_layer_features.size() == torch.Size([1, 27, 768])

# Extract all layers' features
all_layers = gottbert.extract_features(tokens, return_all_hiddens=True)
assert len(all_layers) == 13
assert torch.all(all_layers[-1] == last_layer_features)
```

## Citation

```bibtex
@misc{scheible2020gottbert,
      title={GottBERT: a pure German Language Model},
      author={Raphael Scheible and Fabian Thomczyk and Patric Tippmann and Victor Jaravine and Martin Boeker},
      year={2020},
      eprint={2012.02110},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```