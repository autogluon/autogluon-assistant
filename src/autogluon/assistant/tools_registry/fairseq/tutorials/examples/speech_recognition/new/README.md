Summary: This tutorial demonstrates how to use Flashlight Decoder for speech recognition in fairseq, covering two main implementation scenarios: basic decoding with a pre-trained model and parameter optimization using Ax sweeper. It shows how to integrate KenLM language models with acoustic models for speech recognition tasks, configure decoding parameters, and evaluate on multiple test sets. The tutorial provides command-line examples for running inference with essential parameters including model checkpoints, data paths, lexicon files, and language models, making it valuable for implementing and optimizing speech recognition systems with external language model integration.

# Flashlight Decoder

This script runs decoding for pre-trained speech recognition models.

## Usage

Assuming a few variables:

```bash
checkpoint=<path-to-checkpoint>
data=<path-to-data-directory>
lm_model=<path-to-language-model>
lexicon=<path-to-lexicon>
```

Example usage for decoding a fine-tuned Wav2Vec model:

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

Example usage for using Ax to sweep WER parameters (requires `pip install hydra-ax-sweeper`):

```bash
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
