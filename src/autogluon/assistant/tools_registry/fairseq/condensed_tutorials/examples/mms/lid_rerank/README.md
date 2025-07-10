# Condensed: N-best Re-ranking for Multilingual LID+ASR

Summary: This tutorial demonstrates a multilingual speech recognition enhancement workflow using N-best re-ranking. It covers implementation techniques for both Whisper and MMS models to generate multiple language identification (LID) candidates and corresponding ASR outputs. The tutorial details how to compute external re-ranking features using language models (MaLA), written language identification (NLLB), and acoustic models (MMS-Zeroshot with U-roman). It provides complete code for tuning feature coefficients on development data and applying optimized weights to test data, ultimately producing improved LID and ASR results through this ensemble approach.

*This is a condensed version that preserves essential implementation details and context.*

# N-best Re-ranking for Multilingual LID+ASR

A workflow for improving multilingual speech recognition by considering N-best language identification (LID) predictions, running ASR in multiple languages, and re-ranking results using external features.

## Implementation Workflow

1. Run LID+ASR inference (MMS or Whisper)
2. Compute external re-ranking features
3. Tune feature coefficients on dev set
4. Apply on test set

## 1) LID+ASR Inference

### Data Preparation
```
# Format: one path per line
/path/to/audio1.wav
/path/to/audio2.wav
/path/to/audio3.wav
```

### Whisper Implementation

```bash
# Run LID with N=10 best predictions
python whisper/infer_lid.py --wavs "path/to/wav/list" --dst "path/to/lid/results" \
    --model large-v2 --n 10

# Run ASR using top-N LID predictions
python whisper/infer_asr.py --wavs "path/to/wav/list" \
    --lids "path/to/lid/results"/nbest_lid --dst "path/to/asr/results" \
    --model large-v2
```

### MMS Implementation

```bash
# Format wav list for MMS
python mms/format_wav_list.py --src "/path/to/wav/list" --dst "/path/to/wav/manifest.tsv"

# Run LID with top-k=10
cd "path/to/fairseq/dir"
PYTHONPATH='.' python3 examples/mms/lid/infer.py "path/to/dict/dir" \
    --path "path/to/model" --task audio_classification \
    --infer-manifest "path/to/wav/manifest.tsv" \
    --output-path "path/to/lid/results" --top-k 10

# Split data by language for parallel processing
python mms/split_by_lang.py --wavs_tsv "/path/to/wav/manifest.tsv" \
    --lid_preds "path/to/lid/results"predictions.txt --dst "path/to/data/split"

# Generate language-specific ASR commands
mms/make_parallel_single_runs.py --dump "path/to/data/split" \
    --model "path/to/model" --dst "path/to/asr/results" \
    --fairseq_dir "path/to/fairseq/dir" > run.sh

# Execute ASR (can be parallelized)
. ./run.sh

# Merge results back to original order
python mms/merge_by_run.py --dump "path/to/data/split" --exp "path/to/asr/results"
```

## 2) External Re-ranking Features

### MaLA - Large Language Model
```bash
python mala/infer.py --txt "path/to/asr/results"/nbest_asr_hyp --dst "path/to/lm/results"
```

### NLLB - Written LID Model
```bash
python nllb/infer.py --txt "path/to/asr/results"/nbest_asr_hyp \
    --dst "path/to/wlid/results" --model "path/to/nllb/model"
```

### MMS-Zeroshot - U-roman Acoustic Model
```bash
# U-romanize N-best ASR hypotheses
python mms-zs/uromanize.py --txt "path/to/asr/results"/nbest_asr_hyp \
    --lid "path/to/lid/results"/nbest_lid --dst "path/to/uasr/results" \
    --model "path/to/mms-zeroshot"

# Compute forced alignment score
python mms-zs/falign.py --uroman_txt "path/to/uasr/results"/nbest_asr_hyp_uroman \
    --wav "path/to/wav/list" --dst "path/to/uasr/results" \
    --model "path/to/mms-zeroshot"
```

## 3) Tune Feature Coefficients

```bash
python rerank/tune_coefficients.py \
    --slid "path/to/lid/results"/slid_score \
    --asr "path/to/asr/results"/asr_score \
    --wlid "path/to/wlid/results"/wlid_score \
    --lm "path/to/lm/results"/lm_score \
    --uasr "path/to/uasr/results"/uasr_score \
    --dst "path/to/rerank/results" \
    --ref_lid "ground-truth/lid" \
    --nbest_lid "path/to/lid/results"/nbest_lid \
    --ref_asr "ground-truth/asr" \
    --nbest_asr "path/to/asr/results"/nbest_asr_hyp
```

## 4) Apply on Test Set

```bash
python rerank/rerank.py \
    --slid "path/to/lid/results"/slid_score \
    --asr "path/to/asr/results"/asr_score \
    --wlid "path/to/wlid/results"/wlid_score \
    --lm "path/to/lm/results"/lm_score \
    --uasr "path/to/uasr/results"/uasr_score \
    --dst "path/to/rerank/results" \
    --ref_lid "ground-truth/lid" \
    --nbest_lid "path/to/lid/results"/nbest_lid \
    --ref_asr "ground-truth/asr" \
    --nbest_asr "path/to/asr/results"/nbest_asr_hyp \
    --w "path/to/rerank/results"/best_coefficients
```

**Output files:**
- Re-ranked LID: `"path/to/rerank/results"/reranked_1best_lid`
- Re-ranked ASR: `"path/to/rerank/results"/reranked_1best_asr_hyp`