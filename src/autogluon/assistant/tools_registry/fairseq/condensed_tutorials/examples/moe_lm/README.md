# Condensed: Efficient Large Scale Language Modeling with Mixtures of Experts

Summary: This tutorial covers implementing and using Mixtures of Experts (MoE) language models for efficient large-scale language modeling. It provides access to pre-trained dense models (125M-13B parameters) and MoE models (15B-1.1T parameters), with code examples for model loading, evaluation, and inference. The tutorial demonstrates how to perform zero-shot evaluation on the COPA task, with critical guidance on handling newlines properly during inference (replacing them with EOS tokens). Key functionalities include loading pre-trained models, converting to half-precision, GPU acceleration, calculating log probabilities, and proper text formatting for models that weren't trained on newline characters.

*This is a condensed version that preserves essential implementation details and context.*

# Efficient Large Scale Language Modeling with Mixtures of Experts

## Pre-trained Models

### Dense Models
Models can be run from the `main` branch:

| Model | Layers | Dim | Download |
|-------|--------|-----|----------|
| `dense_125m` | 12 | 768 | [en_dense_lm_125m.tar.gz (0.2GB)](https://dl.fbaipublicfiles.com/fairseq/models/lm/en_dense_lm_125m.tar.gz) |
| `dense_355m` | 24 | 1024 | [en_dense_lm_355m.tar.gz (0.6GB)](https://dl.fbaipublicfiles.com/fairseq/models/lm/en_dense_lm_355m.tar.gz) |
| `dense_1_3b` | 24 | 2048 | [en_dense_lm_1_3b.tar.gz (2.3GB)](https://dl.fbaipublicfiles.com/fairseq/models/lm/en_dense_lm_1_3b.tar.gz) |
| `dense_2_7b` | 32 | 2560 | [en_dense_lm_2_7b.tar.gz (4.6GB)](https://dl.fbaipublicfiles.com/fairseq/models/lm/en_dense_lm_2_7b.tar.gz) |
| `dense_6_7b` | 32 | 4096 | [en_dense_lm_6_7b.tar.gz (12GB)](https://dl.fbaipublicfiles.com/fairseq/models/lm/en_dense_lm_6_7b.tar.gz) |
| `dense_13b` | 40 | 5120 | [en_dense_lm_13b.tar.gz (23GB)](https://dl.fbaipublicfiles.com/fairseq/models/lm/en_dense_lm_13b.tar.gz) |

### MoE Models
MoE models must be run from the `moe` branch:

| Model | Layers | Dim | Download |
|-------|--------|-----|----------|
| `moe_15b` | 12 | 768 | [en_moe_lm_15b.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/lm/en_moe_lm_15b.tar.gz) |
| `moe_52b` | 24 | 1024 | [en_moe_lm_52b.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/lm/en_moe_lm_52b.tar.gz) |
| `moe_207b` | 24 | 2048 | Available by request |
| `moe_1_1t` | 32 | 4096 | Available by request |

## Evaluation

### Example Implementation (COPA)

```python
from fairseq.models.transformer_lm import TransformerLanguageModel
model_dir = '/path/to/en_dense_lm_125m'
lm = TransformerLanguageModel.from_pretrained(model_dir, bpe='gpt2')
lm = lm.eval();  # disable dropout
lm = lm.half();  # use FP16 for evaluation
lm = lm.cuda();  # move to GPU

def get_logprobs(prompt):
    import re
    prompt = re.sub('\n+' , '\n', prompt)  # collapse repeated newlines
    return lm.score(prompt, replace_newlines_with_eos=True)['positional_scores']

# Zero-shot evaluation for COPA task
def COPA_eval(prompt, alternative1, alternative2):
    lprob1 = get_logprobs(prompt + "\n" + alternative1).sum()
    lprob2 = get_logprobs(prompt + "\n" + alternative2).sum()
    return 1 if lprob1 > lprob2 else 2
```

### Important Data Format Considerations

**Critical Warning:** The model never saw newline characters during pretraining. Newlines should be replaced with the end-of-sentence symbol (`</s>`) during few-shot prompting.

During pretraining, data was formatted as:
```
<doc0,para0,tok0> ... <doc0,para0,tokX>
<doc0,para1,tok0> ... <doc0,para1,tokY>

<doc1,para0,tok0> ... <doc0,para0,tokX>
...
```

#### Correct Handling of Newlines

```python
# INCORRECT - will encode actual newlines
tokens_bad = lm.score(data)['tokens']

# CORRECT - use replace_newline_with_eos option
tokens_good = lm.score(data, replace_newline_with_eos=True)['tokens']
```