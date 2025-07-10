# Condensed: Flashlight Decoder

Summary: This tutorial demonstrates how to use Flashlight Decoder for speech recognition in fairseq, covering two main implementation scenarios: basic decoding with a pre-trained model and parameter optimization using Ax sweeper. It shows how to integrate KenLM language models with acoustic models for speech recognition tasks, configure decoding parameters, and evaluate on multiple test sets. The tutorial provides command-line examples for running inference with essential parameters including model checkpoints, data paths, lexicon files, and language models, making it valuable for implementing and optimizing speech recognition systems with external language model integration.

*This is a condensed version that preserves essential implementation details and context.*

# Flashlight Decoder for Speech Recognition

## Usage Examples

### Basic Decoding with Pre-trained Model

```bash
python $FAIRSEQ_ROOT/examples/speech_recognition/new/infer.py --multirun \
    task=audio_pretraining \
    task.data=$data \
    task.labels=ltr \
    common_eval.path=$checkpoint \
    decoding.type=kenlm \
    decoding.lexicon=$lexicon \
    decoding.lmpath=$lm_model \
    dataset.gen_subset=dev_clean,dev_other,test_clean,test_other
```

### Parameter Sweeping with Ax

```bash
# Requires: pip install hydra-ax-sweeper
python $FAIRSEQ_ROOT/examples/speech_recognition/new/infer.py --multirun \
    hydra/sweeper=ax \
    task=audio_pretraining \
    task.data=$data \
    task.labels=ltr \
    common_eval.path=$checkpoint \
    decoding.type=kenlm \
    decoding.lexicon=$lexicon \
    decoding.lmpath=$lm_model \
    dataset.gen_subset=dev_other
```

Key variables:
- `checkpoint`: Path to model checkpoint
- `data`: Path to data directory
- `lm_model`: Path to language model
- `lexicon`: Path to lexicon file