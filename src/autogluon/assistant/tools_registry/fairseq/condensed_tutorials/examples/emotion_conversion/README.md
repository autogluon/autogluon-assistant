# Condensed: Textless speech emotion conversion using decomposed and discrete representations

Summary: This tutorial implements textless speech emotion conversion using deep learning. It covers techniques for converting speech between different emotional styles (neutral, amused, angry, sleepy, disgusted) without text transcription. Key functionalities include: preprocessing audio into discrete tokens using HuBERT, training a pipeline of models (HiFiGAN vocoder, BART-style denoising pretraining, emotion translation model, F0 and duration predictors), and inference for generating emotionally-converted speech. The implementation uses a sequence-to-sequence approach with specialized components for handling prosodic features. This tutorial helps with speech synthesis, emotion conversion, and working with discrete speech representations.

*This is a condensed version that preserves essential implementation details and context.*

# Textless Speech Emotion Conversion Implementation Guide

## Installation

```bash
# Create and activate conda environment
conda create -n emotion python=3.8 -y
conda activate emotion

# Clone repository
git clone https://github.com/facebookresearch/fairseq.git
cd fairseq/examples/emotion_conversion
git clone https://github.com/felixkreuk/speech-resynthesis

# Download EmoV discrete tokens
wget https://dl.fbaipublicfiles.com/textless_nlp/emotion_conversion/data.tar.gz
tar -xzvf data.tar.gz

# Install dependencies
pip install --editable ./
pip install -r examples/emotion_conversion/requirements.txt
```

## Data Preprocessing

1. **Convert audio to discrete representations**
   - Use [HuBERT checkpoint](https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt)
   - Use k-means model at `data/hubert_base_ls960_layer9_clusters200/data_hubert_base_ls960_layer9_clusters200.bin`
   - Follow steps from [HuBERT simple_kmeans](https://github.com/pytorch/fairseq/tree/main/examples/hubert/simple_kmeans)

2. **Create data splits**
   ```bash
   python examples/emotion_conversion/preprocess/create_core_manifest.py \
       --tsv data/data.tsv \
       --emov-km data/hubert_base_ls960_layer9_clusters200/data.km \
       --km data/hubert_base_ls960_layer9_clusters200/vctk.km \
       --dict data/hubert_base_ls960_layer9_clusters200/dict.txt \
       --manifests-dir $DATA
   ```

3. **Extract F0**
   ```bash
   python examples/emotion_conversion/preprocess/extract_f0.py \
       --tsv data/data.tsv \
       --extractor pyaapt
   ```

## Model Training Pipeline

### 1. HiFiGAN Vocoder Training
```bash
python examples/emotion_conversion/speech-resynthesis/train.py \
    --checkpoint_path <hifigan-checkpoint-dir> \
    --config examples/emotion_conversion/speech-resynthesis/configs/EmoV/emov_hubert-layer9-cluster200_fixed-spkr-embedder_f0-raw_gst.json
```

### 2. Translation Pre-training (BART-style denoising)
```bash
python train.py \
    $DATA/fairseq-data/emov_multilingual_denoising_cross-speaker_dedup_nonzeroshot/tokenized \
    --save-dir <your-save-dir> \
    --tensorboard-logdir <your-tb-dir> \
    --langs neutral,amused,angry,sleepy,disgusted,vctk.km \
    --dataset-impl mmap \
    --task multilingual_denoising \
    --arch transformer_small --criterion cross_entropy \
    --multilang-sampling-alpha 1.0 --sample-break-mode eos --max-tokens 16384 \
    --update-freq 1 --max-update 3000000 \
    --dropout 0.1 --attention-dropout 0.1 --relu-dropout 0.0 \
    --optimizer adam --weight-decay 0.01 --adam-eps 1e-06 \
    --clip-norm 0.1 --lr-scheduler polynomial_decay --lr 0.0003 \
    --total-num-update 3000000 --warmup-updates 10000 --fp16 \
    --poisson-lambda 3.5 --mask 0.3 --mask-length span-poisson --replace-length 1 \
    --rotate 0 --mask-random 0.1 --insert 0 --permute-sentences 1.0 \
    --skip-invalid-size-inputs-valid-test \
    --user-dir examples/emotion_conversion/fairseq_models
```

### 3. Emotion Translation Model Training
```bash
python train.py \
    --distributed-world-size 1 \
    $DATA/fairseq-data/emov_multilingual_translation_cross-speaker_dedup/tokenized/ \
    --save-dir <your-save-dir> \
    --tensorboard-logdir <your-tb-dir> \
    --arch multilingual_small --task multilingual_translation \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
    --lang-pairs neutral-amused,neutral-sleepy,neutral-disgusted,neutral-angry,amused-sleepy,amused-disgusted,amused-neutral,amused-angry,angry-amused,angry-sleepy,angry-disgusted,angry-neutral,disgusted-amused,disgusted-sleepy,disgusted-neutral,disgusted-angry,sleepy-amused,sleepy-neutral,sleepy-disgusted,sleepy-angry \
    --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --lr 1e-05 --clip-norm 0 --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --warmup-updates 2000 --lr-scheduler inverse_sqrt \
    --max-tokens 4096 --update-freq 1 --max-update 100000 \
    --required-batch-size-multiple 8 --fp16 --num-workers 4 \
    --seed 2 --log-format json --log-interval 25 --save-interval-updates 1000 \
    --no-epoch-checkpoints --keep-best-checkpoints 1 --keep-interval-updates 1 \
    --finetune-from-model <path-to-model-from-previous-step> \
    --user-dir examples/emotion_conversion/fairseq_models
```

**Key options:**
- `--share-encoders` and `--share-decoders` to share parameters
- `--encoder-langtok {'src'|'tgt'}` and `--decoder-langtok` to add emotion tokens

### 4. F0-Predictor Training
```bash
cd examples/emotion_conversion
python -m emotion_models.pitch_predictor n_tokens=200 \
    train_tsv="$DATA/denoising/emov/train.tsv" \
    train_km="$DATA/denoising/emov/train.km" \
    valid_tsv="$DATA/denoising/emov/valid.tsv" \
    valid_km="$DATA/denoising/emov/valid.km"
```

### 5. Duration-Predictor Training
```bash
cd examples/emotion_conversion
for emotion in "neutral" "amused" "angry" "disgusted" "sleepy"; do
    python -m emotion_models.duration_predictor n_tokens=200 substring=$emotion \
        train_tsv="$DATA/denoising/emov/train.tsv" \
        train_km="$DATA/denoising/emov/train.km" \
        valid_tsv="$DATA/denoising/emov/valid.tsv" \
        valid_km="$DATA/denoising/emov/valid.km"
done
```

## Inference Pipeline

### 1. Token Generation
```bash
fairseq-generate \
    $DATA/fairseq-data/emov_multilingual_translation_cross-speaker_dedup/tokenized/ \
    --task multilingual_translation \
    --gen-subset test \
    --path <your-saved-translation-checkpoint> \
    --beam 5 \
    --batch-size 4 --max-len-a 1.8 --max-len-b 10 --lenpen 1 --min-len 1 \
    --skip-invalid-size-inputs-valid-test --distributed-world-size 1 \
    --source-lang neutral --target-lang amused \
    --lang-pairs neutral-amused,neutral-sleepy,neutral-disgusted,neutral-angry,amused-sleepy,amused-disgusted,amused-neutral,amused-angry,angry-amused,angry-sleepy,angry-disgusted,angry-neutral,disgusted-amused,disgusted-sleepy,disgusted-neutral,disgusted-angry,sleepy-amused,sleepy-neutral,sleepy-disgusted,sleepy-angry \
    --results-path <token-output-path> \
    --user-dir examples/emotion_conversion/fairseq_models
```

### 2. Waveform Synthesis
```bash
python examples/emotion_conversion/synthesize.py \
    --result-path <token-output-path>/generate-test.txt \
    --data $DATA/fairseq-data/emov_multilingual_translation_cross-speaker_dedup/neutral-amused \
    --orig-tsv examples/emotion_conversion/data/data.tsv \
    --orig-km examples/emotion_conversion/data/hubert_base_ls960_layer9_clusters200/data.km \
    --checkpoint-file <hifigan-checkpoint-dir>/g_00400000 \
    --dur-model duration_predictor/ \
    --f0-model pitch_predictor/pitch_predictor.ckpt \
    -s neutral -t amused \
    --outdir ~/tmp/emotion_results/wavs/neutral-amused
```

**Important:** Ensure source and target emotions match between token generation and waveform synthesis steps.

## Citation
```
@article{kreuk2021textless,
  title={Textless speech emotion conversion using decomposed and discrete representations},
  author={Kreuk, Felix and Polyak, Adam and Copet, Jade and Kharitonov, Eugene and Nguyen, Tu-Anh and Rivi{\`e}re, Morgane and Hsu, Wei-Ning and Mohamed, Abdelrahman and Dupoux, Emmanuel and Adi, Yossi},
  journal={Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2022}
}
```