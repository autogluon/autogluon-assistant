# Condensed: Self-Training with Kaldi HMM Models

Summary: This tutorial demonstrates implementing self-training with Kaldi HMM models for speech recognition. It covers techniques for decoding speech into phones or words using pre-trained models, with specific implementation details for model training, evaluation, and selection using unsupervised metrics. The tutorial helps with tasks like configuring Kaldi for speech decoding, preparing WFSTs for word-level transcription, and evaluating model performance through metrics like WER and perplexity. Key functionalities include phone decoding, word decoding (via a two-step process), and unsupervised model selection based on combined metrics of error rates and language model perplexity.

*This is a condensed version that preserves essential implementation details and context.*

# Self-Training with Kaldi HMM Models

This guide covers implementing self-training using Kaldi HMM models for decoding into phones or words.

## Prerequisites
- Install Kaldi and place this folder in `path/to/kaldi/egs`
- Prepare:
  - `w2v_dir`: features (`{train,valid}.{npy,lengths}`), transcripts (`{train,valid}.${label}`), and dictionary (`dict.${label}.txt`)
  - `lab_dir`: pseudo labels (`{train,valid}.txt`)
  - `arpa_lm`: n-gram phone LM for decoding
  - `arpa_lm_bin`: n-gram phone LM for unsupervised model selection (KenLM compatible)

## Training Process

1. Configure `train.sh` with the required variables and output directory
2. Run the script to train HMM models

```bash
# Example output from training
INFO:root:./out/exp/mono/decode_valid/scoring/14.0.0.tra.txt: score 0.9178 wer 28.71% lm_ppl 24.4500 gt_wer 25.57%
INFO:root:./out/exp/tri1/decode_valid/scoring/17.1.0.tra.txt: score 0.9257 wer 26.99% lm_ppl 30.8494 gt_wer 21.90%
INFO:root:./out/exp/tri2b/decode_valid/scoring/8.0.0.tra.txt: score 0.7506 wer 23.15% lm_ppl 25.5944 gt_wer 15.78%
```

**Key metrics:**
- `wer`: Word error rate vs pseudo labels
- `gt_wer`: Word error rate vs ground truth
- `lm_ppl`: Language model perplexity
- `score`: Unsupervised metric for model selection (lower is better)

## Phone Decoding

Configure `decode_phone.sh`:
- `out_dir`: Same as training
- `dec_exp`: Selected model (e.g., `tri2b`)
- `dec_lmparam`: Selected LM parameter (e.g., `8.0.0`)
- `dec_script`: Use `decode.sh` for mono/tri1/tri2b; `decode_fmllr.sh` for tri3b

Output will be saved to `out_dir/dec_data`

## Word Decoding

### Step 1: Prepare WFSTs
Configure `decode_word_step1.sh` with:
- All previous variables
- `wrd_arpa_lm`: n-gram word LM for decoding
- `wrd_arpa_lm_bin`: n-gram word LM for unsupervised model selection

```bash
# Example output
INFO:root:./out/exp/tri2b/decodeword_valid/scoring/17.0.0.tra.txt: score 1.8693 wer 24.97% lm_ppl 1785.5333 gt_wer 31.45%
```

### Step 2: Decode into Words
- Set the selected LM parameter in `decode_word_step2.sh`
- Run the script
- Output will be saved to `out_dir/dec_data_word`