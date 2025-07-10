# Condensed: ASR-based evaluation

Summary: This tutorial explains how to evaluate Unit Language Models (ULMs) using ASR-based metrics. It covers the implementation process for: preprocessing audio samples (downsampling to 16kHz and matching audio lengths), preparing manifest files, running ASR with KenLM and Flashlight decoder, and calculating evaluation metrics including Perplexity, Self-BLEU, Auto-BLEU, Continuation-BLEU, and AUC. The tutorial provides complete bash commands for each step, making it valuable for tasks involving speech generation evaluation, audio preprocessing, ASR integration, and quantitative assessment of speech quality, fluency and diversity.

*This is a condensed version that preserves essential implementation details and context.*

# ASR-based Evaluation for ULM

## Evaluation Process Overview
1. Train ULM and sample from it
2. Run UTS on sampled unit sequences
3. Pre-process for ASR
4. Run ASR
5. Calculate post-ASR evaluation metrics

## Preprocessing

### Down-sampling to 16KHz
```bash
python $FAIRSEQ_ROOT/examples/textless_nlp/gslm/unit2speech/convert_to_16k.py $UTS_OUTPUT $UTS_OUTPUT_DOWNSAMPLE
```

### Matching Audio Lengths (Optional)
Important for comparing fluency and diversity with ground-truth speech:
```bash
python $FAIRSEQ_ROOT/examples/textless_nlp/asr_metrics/cut_as.py \
    --samples_dir=$UTS_OUTPUT_DOWNSAMPLE --out_dir=$UTS_OUTPUT_DOWNSAMPLE_CUT \
    --prompts_description=data/ground_truth_continuation_dev.json
```
- Ground-truth files: [dev-clean](https://dl.fbaipublicfiles.com/textless_nlp/gslm/eval_data/ground_truth_continuation_dev.json) and [test-clean](https://dl.fbaipublicfiles.com/textless_nlp/gslm/eval_data/ground_truth_continuation_test.json)
- Contains texts for audio sequences ≥ 6s long

## Running ASR

### Prepare Manifest Files
```bash
python $FAIRSEQ_ROOT/examples/wav2vec/wav2vec_manifest.py \
    $UTS_OUTPUT_DOWNSAMPLE_CUT --valid-percent 0.0 --dest $MANIFEST_DIR --ext wav
```

### Generate Dummy Transcripts
```bash
cp $FAIRSEQ_ROOT/examples/textless_nlp/gslm/asr_metrics/misc/dict.ltr.txt $MANIFEST_DIR
python $FAIRSEQ_ROOT/examples/textless_nlp/gslm/asr_metrics/misc/dummy_asr_data.py --tsv=$MANIFEST_DIR/train.tsv \
 --output-dir=$MANIFEST_DIR
```

### Execute ASR
```bash
mkdir -p asr
python $FAIRSEQ_ROOT/examples/speech_recognition/infer.py \
    $MANIFEST_DIR \
    --task audio_pretraining --nbest 1 --path 960h_scratch.pt \
    --gen-subset=train --results-path $PATH_TO_ASR_OUTPUT \
    --w2l-decoder kenlm --lm-model 4-gram.bin \
    --lexicon librispeech/lexicon_ltr.lst --word-score -1 \
    --sil-weight 0 --lm-weight 2 --criterion ctc --labels ltr --max-tokens 300000 --remove-bpe letter
```
- Requires KenLM, Flashlight decoder, and KenLM 4-gram English language model
- LibriSpeech lexicon available [here](https://dl.fbaipublicfiles.com/textless_nlp/gslm/eval_data/lexicon_ltr.lst)

## Evaluation Metrics
Evaluation runs on the 1,000 shortest sequences ≥ 6s long.

### Perplexity (PPX)
```bash
python ppx.py $PATH_TO_ASR_OUTPUT/hypo.word-960h_scratch.pt-train.txt --cut-tail \
  --manifest=$MANIFEST_DIR/train.tsv --prompts-description=data/ground_truth_continuation_dev.json
```
- `--cut-tail` ignores the last token on each line (ASR sequence ID)

### Self- and Auto-BLEU
```bash
python self_bleu.py $PATH_TO_ASR_OUTPUT/hypo.word-960h_scratch.pt-train.txt --cut-tail \
  --manifest=$MANIFEST_DIR/train.tsv --prompts-description=data/ground_truth_continuation_dev.json
```

### Continuation-BLEU
```bash
python continuation_eval.py --asr-transcript $PATH_TO_ASR_OUTPUT/hypo.word-960h_scratch.pt-train.txt \
   --manifest=$MANIFEST_DIR/train.tsv --prompts-description=data/ground_truth_continuation_dev.json
```

### AUC
Calculate the AUC of perplexity/diversity trade-off using the above metrics. Example in [this Colab notebook](https://colab.research.google.com/drive/1pVPfOVax_PU3MkYdHRSsa-SI8GBUldNt?usp=sharing).