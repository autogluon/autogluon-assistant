# Condensed: BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension

Summary: This tutorial demonstrates implementing BART, a sequence-to-sequence pre-trained model, in PyTorch. It covers loading pre-trained BART models (base, large, and task-specific variants), text processing with BPE encoding/decoding, feature extraction, classification tasks with pre-trained heads, batched prediction, mask filling capabilities, and evaluation for MNLI and summarization tasks. Key functionalities include using BART for text classification, sequence generation, feature extraction, and masked language modeling. The tutorial provides code examples for both CPU and GPU implementations, handling batched inputs, and best practices for inference, making it valuable for developers implementing NLP tasks with transformer-based models.

*This is a condensed version that preserves essential implementation details and context.*

# BART: Denoising Sequence-to-Sequence Pre-training

## Pre-trained Models

| Model | Description | # params | 
|---|---|---|
| `bart.base` | 6 encoder/decoder layers | 140M |
| `bart.large` | 12 encoder/decoder layers | 400M |
| `bart.large.mnli` | Finetuned on MNLI | 400M |
| `bart.large.cnn` | Finetuned on CNN-DM | 400M |
| `bart.large.xsum` | Finetuned on Xsum | 400M |

## Implementation

### Loading BART

```python
# From torch.hub (PyTorch >= 1.1)
import torch
bart = torch.hub.load('pytorch/fairseq', 'bart.large')
bart.eval()  # disable dropout

# For PyTorch 1.0 or custom models
from fairseq.models.bart import BARTModel
bart = BARTModel.from_pretrained('/path/to/bart.large', checkpoint_file='model.pt')
bart.eval()
```

### Text Processing and Feature Extraction

```python
# Apply BPE to input text
tokens = bart.encode('Hello world!')
assert tokens.tolist() == [0, 31414, 232, 328, 2]
bart.decode(tokens)  # 'Hello world!'

# Extract features
last_layer_features = bart.extract_features(tokens)
assert last_layer_features.size() == torch.Size([1, 5, 1024])

# Extract all layers' features
all_layers = bart.extract_features(tokens, return_all_hiddens=True)
assert len(all_layers) == 13
```

### Classification Tasks

```python
# Load MNLI-finetuned model
bart = torch.hub.load('pytorch/fairseq', 'bart.large.mnli')
bart.eval()

# Sentence-pair classification
tokens = bart.encode('BART is a seq2seq model.', 'BART is not sequence to sequence.')
bart.predict('mnli', tokens).argmax()  # 0: contradiction

# Register new classification head
bart.register_classification_head('new_task', num_classes=3)
logprobs = bart.predict('new_task', tokens)
```

### Batched Prediction

```python
import torch
from fairseq.data.data_utils import collate_tokens

batch_of_pairs = [
    ['BART is a seq2seq model.', 'BART is not sequence to sequence.'],
    ['BART is denoising autoencoder.', 'BART is version of autoencoder.'],
]

batch = collate_tokens(
    [bart.encode(pair[0], pair[1]) for pair in batch_of_pairs], pad_idx=1
)

logprobs = bart.predict('mnli', batch)
print(logprobs.argmax(dim=1))  # tensor([0, 2])

# Using GPU
bart.cuda()
```

### Filling Masks

```python
bart = torch.hub.load('pytorch/fairseq', 'bart.base')
bart.eval()

# Default: enforce output length matches input
bart.fill_mask(['The cat <mask> on the <mask>.'], topk=3, beam=10)
# [('The cat was on the ground.', tensor(-0.6183)), ...]

# Allow different output length
bart.fill_mask(['The cat <mask> on the <mask>.'], 
               topk=3, beam=10, match_source_len=False)

# Batch processing with GPU
bart.cuda()
bart.fill_mask(['The cat <mask> on the <mask>.', 
                'The dog <mask> on the <mask>.'], topk=3, beam=10)
```

## Evaluation

### MNLI Evaluation

```python
label_map = {0: 'contradiction', 1: 'neutral', 2: 'entailment'}
ncorrect, nsamples = 0, 0
bart.cuda()
bart.eval()
with open('glue_data/MNLI/dev_matched.tsv') as fin:
    fin.readline()
    for index, line in enumerate(fin):
        tokens = line.strip().split('\t')
        sent1, sent2, target = tokens[8], tokens[9], tokens[-1]
        tokens = bart.encode(sent1, sent2)
        prediction = bart.predict('mnli', tokens).argmax().item()
        prediction_label = label_map[prediction]
        ncorrect += int(prediction_label == target)
        nsamples += 1
        print('| Accuracy: ', float(ncorrect)/float(nsamples))
# Expected: 0.9010
```

### Summarization Evaluation

```bash
# Generate summaries
python examples/bart/summarize.py \
  --model-dir pytorch/fairseq \
  --model-file bart.large.cnn \
  --src cnn_dm/test.source \
  --out cnn_dm/test.hypo

# Calculate ROUGE scores
export CLASSPATH=/path/to/stanford-corenlp-full-2016-10-31/stanford-corenlp-3.7.0.jar
cat test.hypo | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > test.hypo.tokenized
cat test.target | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > test.hypo.target
files2rouge test.hypo.tokenized test.hypo.target
# Expected: ROUGE-2 Average_F: 0.21238
```

## Best Practices
- Use `bart.eval()` for inference to disable dropout
- For summarization tasks, use the specialized models (`bart.large.cnn`, `bart.large.xsum`)
- For classification, consider using `bart.large.mnli` as starting point
- GPU usage is recommended for batch processing